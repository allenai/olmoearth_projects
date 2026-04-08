"""Test raster smoothing and polygon extraction.

Creates synthetic GeoTIFF rasters across 10 quarterly timesteps with four
regions:
1. Always class 2: present every timestep -> should produce a polygon.
2. False positive: class 2 in only 1 timestep -> should be smoothed away.
3. Appears late: class 2 in last 5 timesteps -> should produce a polygon.
4. Hole: class 2 in all but 1 timestep -> hole should be filled, polygon present.

All other pixels are class 1 (background).
"""

import json
import pathlib
from collections.abc import Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from upath import UPath

from olmoearth_projects.projects.satlas.smooth_rasters import (
    smooth_rasters_and_extract_polygons,
)

# 10 quarterly labels.
LABELS = [
    "2022-01",
    "2022-04",
    "2022-07",
    "2022-10",
    "2023-01",
    "2023-04",
    "2023-07",
    "2023-10",
    "2024-01",
    "2024-04",
]

TILE_SIZE = 100
TILE_FNAME = "test_tile.tif"
# UTM zone 1N, 10m resolution covering a small area.
TILE_CRS = CRS.from_epsg(32601)
# Affine transform: 10m pixel, origin at (500000, 1000000) in UTM coords.
TILE_TRANSFORM = from_bounds(
    500000,
    1000000 - TILE_SIZE * 10,
    500000 + TILE_SIZE * 10,
    1000000,
    TILE_SIZE,
    TILE_SIZE,
)


def _write_geotiff(path: pathlib.Path, data: np.ndarray) -> None:
    """Write a single-band uint8 GeoTIFF with CRS and transform."""
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        crs=TILE_CRS,
        transform=TILE_TRANSFORM,
    ) as dst:
        dst.write(data, 1)


def test_smooth_rasters_four_regions(tmp_path: pathlib.Path) -> None:
    """Test raster smoothing with four regions of varying temporal patterns."""
    historical_dir = UPath(tmp_path / "historical")
    smoothed_dir = UPath(tmp_path / "smoothed")

    n_ts = len(LABELS)

    # Build all timesteps as a single (T, H, W) array.
    rasters = np.ones((n_ts, TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    # Region 1 (always present): rows 5-15, cols 5-15.
    rasters[:, 5:15, 5:15] = 2

    # Region 2 (false positive): rows 5-15, cols 60-70. Only at index 3.
    rasters[3, 5:15, 60:70] = 2

    # Region 3 (appears late): rows 60-70, cols 5-15. Last 5 timesteps.
    rasters[5:, 60:70, 5:15] = 2

    # Region 4 (hole): rows 60-70, cols 60-70. All except index 4.
    rasters[:, 60:70, 60:70] = 2
    rasters[4, 60:70, 60:70] = 1  # punch the hole

    # Write each timestep as a GeoTIFF.
    for label_idx, label in enumerate(LABELS):
        label_dir = historical_dir / label
        label_dir.mkdir(parents=True)
        _write_geotiff(
            pathlib.Path(str(label_dir / TILE_FNAME)),
            rasters[label_idx],
        )

    # Run smoothing.
    smooth_rasters_and_extract_polygons(
        application_name="solar_farm",
        label="2024-04",
        historical_dir=str(historical_dir),
        smoothed_dir=str(smoothed_dir),
        workers=1,
    )

    # Read the output GeoJSON.
    out_path = smoothed_dir / "2024-04" / "test_tile.geojson"
    assert out_path.exists(), f"Output GeoJSON should exist at {out_path}"
    with open(out_path) as f:
        fc = json.load(f)

    features = fc["features"]

    # All valid polygons should have value=2.
    assert all(f["properties"]["value"] == 2 for f in features), (
        "All polygons should be class 2"
    )

    # Region 1 (always): should have polygon.
    # Region 2 (false positive): should NOT have polygon (1/10 too sparse).
    # Region 3 (appears late): should have polygon (5/10 is enough).
    # Region 4 (hole): should have polygon (9/10 with hole filled).
    #
    # Expect at least 2 polygons (always + hole), at most 4.
    # The false positive should NOT survive.
    assert len(features) >= 2, (
        f"Expected at least 2 polygons (always + hole), got {len(features)}"
    )
    assert len(features) <= 4, (
        f"Expected at most 4 polygons, got {len(features)}. "
        "False positive region should be smoothed away."
    )

    # Verify coordinates are in WGS84 (longitude/latitude range).
    for feat in features:
        coords = feat["geometry"]["coordinates"]
        # Flatten nested coordinate lists to check ranges.
        flat = _flatten_coords(coords)
        for lon, lat in flat:
            assert -180 <= lon <= 180, f"Longitude {lon} out of WGS84 range"
            assert -90 <= lat <= 90, f"Latitude {lat} out of WGS84 range"


def _flatten_coords(coords: Sequence[int | float]) -> Sequence[Sequence[int | float]]:
    """Recursively flatten nested coordinate lists to (lon, lat) pairs."""
    if isinstance(coords[0], int | float):
        return [coords[:2]]
    result = []
    for item in coords:
        result.extend(_flatten_coords(item))
    return result
