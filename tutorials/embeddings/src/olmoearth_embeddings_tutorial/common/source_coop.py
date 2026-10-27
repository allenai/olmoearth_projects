"""Read OlmoEarth v1.3 global annual embeddings from Source Cooperative.

The embeddings are a public GeoZarr store with one group per UTM zone
(``utm01`` to ``utm60``). Each group holds an int8 array of shape
``(year, 128, y, x)`` on a 10-meter grid in that zone's northern UTM CRS.
See https://source.coop/olmoearth/olmoearth-v1-3-embeddings for details.

The archive, geometry, and parity helpers follow the Python that the
Embeddings Explorer generates (https://olmoearth.allenai.org/embeddings), so
reads here match what the explorer shows. :func:`fetch_embeddings` adds two
things for larger areas: it reads chunk-aligned tiles rather than the whole
area at once, and it averages the parity-corrected 10-meter vectors down to a
coarser output resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import zarr
import zarr.storage
from obstore.store import S3Store
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as transform_geometry

from olmoearth_embeddings_tutorial.common.embedding_utils import (
    NODATA_EMB,
    dequantize,
    quantize,
)

STORE_URL = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
STORE_ENDPOINT = "https://data.source.coop"

# Archive layout details
EMBEDDINGS_ARRAY = "embeddings"
NODATA = NODATA_EMB
CHUNK_SIZE = 256
GRID_WIDTH = 98304
GRID_HEIGHT = 1867776
GRID_TRANSFORM = Affine(10, 0, 0, 0, -10, 9502720)
NATIVE_RESOLUTION_M = 10

PARITY_SAMPLE_CELLS = 16384
PARITY_MIN_PIXELS = 64

READ_TILE_PX = 1024
"""Side length of each 10-meter read, a multiple of CHUNK_SIZE."""

STRIP_ROWS = 256
"""Rows dequantized at a time while averaging, to bound memory use."""


def open_archive(store_url: str = STORE_URL, endpoint: str = STORE_ENDPOINT) -> Any:
    """Open the public embeddings store anonymously."""
    store = S3Store.from_url(store_url, endpoint=endpoint, skip_signature=True)
    return zarr.open_group(zarr.storage.ObjectStore(store, read_only=True), mode="r")


# --------------------
# Zone and geometry helpers (as in the Embeddings Explorer's Python export)
# --------------------

LONGITUDE_ORIGIN = -180
UTM_ZONE_WIDTH = 6
SOUTHERN_LATITUDE_LIMIT = -90
NORTHERN_LATITUDE_LIMIT = 90


def utm_zone(lon: float) -> int:
    """Return the UTM zone number (1 to 60) containing *lon*."""
    return min(60, int((lon - LONGITUDE_ORIGIN) % 360 // UTM_ZONE_WIDTH) + 1)


def zone_group(zone: int) -> str:
    """Return the store group name for a UTM zone."""
    return f"utm{zone:02d}"


def zones_for_aoi(rings: list[list[list[float]]]) -> list[int]:
    """Return the UTM zones an area of interest touches, west to east."""
    longitudes = [lon for ring in rings for lon, _ in ring]
    if not longitudes:
        return []
    first = utm_zone(min(longitudes))
    last = utm_zone(max(longitudes))
    if first <= last:
        return list(range(first, last + 1))
    return list(range(first, 61)) + list(range(1, last + 1))


def zone_band(zone: int, aoi_west: float, aoi_east: float) -> tuple[float, float]:
    """Return the longitude band of *zone*, unwrapped to sit near the AOI."""
    west = LONGITUDE_ORIGIN + (zone - 1) * UTM_ZONE_WIDTH
    middle = (aoi_west + aoi_east) / 2
    west += round((middle - west - UTM_ZONE_WIDTH / 2) / 360) * 360
    return west, west + UTM_ZONE_WIDTH


def zone_longitude_bounds(
    zone: int, aoi_west: float, aoi_east: float, zones: list[int]
) -> tuple[float, float]:
    """Return the longitude range of the AOI that belongs to *zone*."""
    west, east = zone_band(zone, aoi_west, aoi_east)
    if zone == zones[0]:
        west = min(west, aoi_west)
    if zone == zones[-1]:
        east = max(east, aoi_east)
    return west, east


def aoi_geometry(
    rings: list[list[list[float]]], zone: int, zones: list[int]
) -> tuple[Any, Transformer]:
    """Return the AOI's share of *zone* in that zone's UTM CRS."""
    crs = CRS.from_proj4(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs")
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    polygon = Polygon(rings[0])
    for ring in rings[1:]:
        polygon = polygon.symmetric_difference(Polygon(ring))
    west, east = zone_longitude_bounds(
        zone, polygon.bounds[0], polygon.bounds[2], zones
    )
    clipped = polygon.intersection(
        box(west, SOUTHERN_LATITUDE_LIMIT, east, NORTHERN_LATITUDE_LIMIT)
    )
    if clipped.is_empty or clipped.area == 0:
        return None, to_utm
    return transform_geometry(to_utm.transform, clipped), to_utm


def align_to_chunks(window: Any) -> tuple[int, int, int, int]:
    """Expand a pixel window outward to whole store chunks."""
    x0 = max(0, int(np.floor(window.col_off / CHUNK_SIZE) * CHUNK_SIZE))
    y0 = max(0, int(np.floor(window.row_off / CHUNK_SIZE) * CHUNK_SIZE))
    x1 = min(
        GRID_WIDTH,
        int(np.ceil((window.col_off + window.width) / CHUNK_SIZE) * CHUNK_SIZE),
    )
    y1 = min(
        GRID_HEIGHT,
        int(np.ceil((window.row_off + window.height) / CHUNK_SIZE) * CHUNK_SIZE),
    )
    return x0, y0, x1, y1


def year_index_for(years: np.ndarray, year: int) -> int:
    """Return the index of *year* on the store's time axis."""
    matches = np.flatnonzero(np.asarray(years).astype(int) == year)
    if not len(matches):
        available = ", ".join(str(value) for value in years)
        raise ValueError(f"Year {year} is unavailable. Available years: {available}")
    return int(matches[0])


# --------------------
# Parity correction (as in the Embeddings Explorer's Python export)
# --------------------


# The archive's encoder leaves a fixed offset per (row % 2, column % 2) grid
# position, which draws one-pixel stripes. Its size varies by region, so it is
# estimated per area.
def parity_offsets(embeddings: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Estimate the per-parity offsets of an int8 ``(C, H, W)`` block."""
    bands, height, width = embeddings.shape
    cells_down, cells_across = height // 2, width // 2
    step = max(
        1, int(np.ceil(np.sqrt(cells_down * cells_across / PARITY_SAMPLE_CELLS)))
    )
    means = np.zeros((2, 2, bands), dtype=np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            sample = embeddings[
                :, dy : cells_down * 2 : step * 2, dx : cells_across * 2 : step * 2
            ]
            sample = sample.reshape(bands, -1)
            sample = sample[:, sample[0] != NODATA]
            if sample.shape[1] < PARITY_MIN_PIXELS:
                return np.zeros((2, 2, bands), dtype=np.float32)
            means[(y0 + dy) % 2, (x0 + dx) % 2] = dequantize(sample).mean(axis=1)
    return means - means.mean(axis=(0, 1))


def corrected_vectors(
    embeddings: np.ndarray, x0: int, y0: int, offsets: np.ndarray | None = None
) -> np.ndarray:
    """Dequantize an int8 block and remove its parity offsets.

    *offsets* defaults to an estimate from *embeddings* itself; pass offsets
    estimated from a larger block to correct it in pieces.
    """
    if offsets is None:
        offsets = parity_offsets(embeddings, x0, y0)
    vectors = dequantize(embeddings)
    for dy in (0, 1):
        for dx in (0, 1):
            vectors[:, dy::2, dx::2] -= offsets[(y0 + dy) % 2, (x0 + dx) % 2][
                :, None, None
            ]
    return vectors


# --------------------
# Download
# --------------------


def _average_tile(
    tile: np.ndarray, inside: np.ndarray, x0: int, y0: int, factor: int
) -> np.ndarray:
    """Parity-correct a 10-meter int8 tile and average *factor* x *factor* blocks.

    Pixels outside the AOI or without data are ignored; a block with no valid
    pixels becomes nodata. Returns int8 in the archive's quantization.
    """
    bands, height, width = tile.shape
    offsets = parity_offsets(tile, x0, y0)
    out = np.full((bands, height // factor, width // factor), NODATA, dtype=np.int8)
    for row in range(0, height, STRIP_ROWS):
        strip = tile[:, row : row + STRIP_ROWS]
        strip_h = strip.shape[1]
        valid = inside[row : row + strip_h] & (strip[0] != NODATA)
        vectors = corrected_vectors(strip, x0, y0 + row, offsets)
        vectors[:, ~valid] = 0.0
        shape = (bands, strip_h // factor, factor, width // factor, factor)
        sums = vectors.reshape(shape).sum(axis=(2, 4))
        counts = valid.reshape(shape[1:]).sum(axis=(1, 3))
        block = quantize(sums / np.maximum(counts, 1))
        block[:, counts == 0] = NODATA
        out[:, row // factor : (row + strip_h) // factor] = block
    return out


def fetch_embeddings(
    geojson: dict[str, Any],
    year: int,
    out_path: Path,
    resolution_m: int = 40,
    dims: int = 128,
) -> Path:
    """Download annual embeddings for an area of interest as an int8 GeoTIFF.

    Args:
        geojson: FeatureCollection with one polygon in WGS 84. The area must
            lie within a single UTM zone.
        year: Reference year of the annual embeddings (2017 to 2025, depending
            on the region).
        out_path: Destination GeoTIFF.
        resolution_m: Output resolution in meters: 10, 20, 40, or 80. Coarser
            output averages the 10-meter vectors, which keeps files small.
        dims: Number of embedding dimensions to read: 128, or 64 for the
            cheaper matryoshka prefix.

    Returns:
        *out_path*.
    """
    factor = resolution_m // NATIVE_RESOLUTION_M
    if resolution_m % NATIVE_RESOLUTION_M or CHUNK_SIZE % factor:
        raise ValueError("resolution_m must be 10, 20, 40, or 80.")

    rings = geojson["features"][0]["geometry"]["coordinates"]
    zones = zones_for_aoi(rings)
    if len(zones) != 1:
        raise ValueError("The area of interest must lie within a single UTM zone.")
    (zone,) = zones
    utm_aoi, _ = aoi_geometry(rings, zone, zones)
    if utm_aoi is None:
        raise ValueError("The area of interest is empty.")

    # Read whole store chunks, but write only the AOI's bounds, snapped
    # outward to the output resolution.
    read_x0, read_y0, read_x1, read_y1 = align_to_chunks(
        from_bounds(*utm_aoi.bounds, transform=GRID_TRANSFORM)
    )
    aoi_window = from_bounds(*utm_aoi.bounds, transform=GRID_TRANSFORM)
    out_x0 = int(np.floor(aoi_window.col_off / factor) * factor)
    out_y0 = int(np.floor(aoi_window.row_off / factor) * factor)
    out_x1 = int(np.ceil((aoi_window.col_off + aoi_window.width) / factor) * factor)
    out_y1 = int(np.ceil((aoi_window.row_off + aoi_window.height) / factor) * factor)

    group = open_archive()[zone_group(zone)]
    year_index = year_index_for(group["time"][:], year)
    array = group[EMBEDDINGS_ARRAY]

    out_h = (out_y1 - out_y0) // factor
    out_w = (out_x1 - out_x0) // factor
    read_gb = (read_y1 - read_y0) * (read_x1 - read_x0) * dims / 1e9
    print(
        f"Source Coop: {zone_group(zone)}, {year}, {out_w}x{out_h} px at "
        f"{resolution_m} m (reading {read_gb:.1f} GB at 10 m)"
    )
    averaged = np.full(
        (dims, (read_y1 - read_y0) // factor, (read_x1 - read_x0) // factor),
        NODATA,
        dtype=np.int8,
    )
    tiles = [
        (y, x)
        for y in range(read_y0, read_y1, READ_TILE_PX)
        for x in range(read_x0, read_x1, READ_TILE_PX)
    ]
    for i, (y, x) in enumerate(tiles, start=1):
        y_end = min(y + READ_TILE_PX, read_y1)
        x_end = min(x + READ_TILE_PX, read_x1)
        tile = np.asarray(array[year_index, :dims, y:y_end, x:x_end], dtype=np.int8)
        inside = geometry_mask(
            [mapping(utm_aoi)],
            out_shape=(y_end - y, x_end - x),
            transform=GRID_TRANSFORM * Affine.translation(x, y),
            invert=True,
        )
        averaged[
            :,
            (y - read_y0) // factor : (y_end - read_y0) // factor,
            (x - read_x0) // factor : (x_end - read_x0) // factor,
        ] = _average_tile(tile, inside, x, y, factor)
        print(f"  tile {i}/{len(tiles)}")

    out = averaged[
        :,
        (out_y0 - read_y0) // factor : (out_y1 - read_y0) // factor,
        (out_x0 - read_x0) // factor : (out_x1 - read_x0) // factor,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "int8",
        "count": dims,
        "height": out_h,
        "width": out_w,
        "crs": rasterio.crs.CRS.from_epsg(32600 + zone),
        "transform": GRID_TRANSFORM
        * Affine.translation(out_x0, out_y0)
        * Affine.scale(factor),
        "nodata": NODATA,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "interleave": "band",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out)
    print(f"  saved {out_path}")
    return out_path
