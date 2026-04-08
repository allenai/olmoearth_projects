"""Raster smoothing and polygon extraction for Satlas segmentation tasks.

Applies temporal Viterbi smoothing per-tile across historical timesteps, then
vectorizes the smoothed rasters into polygons (in WGS84 GeoJSON).

Each tile is a GeoTIFF covering a degree-based region (e.g., 5x5 degrees).
The GeoTIFF may be in any CRS (typically UTM).
"""

import json
import multiprocessing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.ops
import torch
import tqdm
from pyproj import Transformer
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_projects.utils.logging import get_logger

logger = get_logger(__name__)

# How many historical timesteps to use for smoothing.
NUM_HISTORICAL_TIMESTEPS = 12

# Minimum area in pixels to consider as a valid polygon.
MIN_AREA = 8


@dataclass
class RasterSmoothConfig:
    """Configuration for HMM-based raster smoothing."""

    num_classes: int
    transition_probs: npt.NDArray
    emission_probs: npt.NDArray
    initial_probs: npt.NDArray
    # If true, only smooth pixels that have class >= 2 at some point.
    # This speeds things up if the foreground class is rare, but will slow things down
    # if most pixels are foreground.
    sparse: bool = False


# Per-application smoothing configs.
RASTER_SMOOTH_CONFIGS: dict[str, RasterSmoothConfig] = {
    "solar_farm": RasterSmoothConfig(
        num_classes=3,
        # Transition: solar farms are rarely destroyed (2% revert) but more
        # likely to be newly built (10% from non-solar to solar).
        transition_probs=np.array(
            [[1, 0, 0], [0, 0.9, 0.1], [0, 0.02, 0.98]], dtype=np.float32
        ),
        # Emission: assume ~20% chance the model confuses the two non-background
        # classes with each other.
        emission_probs=np.array(
            [[1, 0, 0], [0.05, 0.75, 0.2], [0.05, 0.2, 0.75]], dtype=np.float32
        ),
        initial_probs=np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32),
        sparse=True,
    ),
}


def _smooth_single_tile(
    tile_fname: str,
    historical_dir: str,
    smoothed_dir: str,
    label: str,
    application_name: str,
) -> None:
    """Smooth one tile across historical timesteps and extract polygons.

    Reads CRS and affine transform from the GeoTIFF header, applies Viterbi
    smoothing, vectorizes the result, and writes a WGS84 GeoJSON.

    Args:
        tile_fname: the tile filename (e.g. "-10_-5_-5_0.tif").
        historical_dir: directory with per-label subdirectories of prediction GeoTIFFs.
        smoothed_dir: directory to write smoothed outputs.
        label: current label (YYYY-MM).
        application_name: application name for config lookup.
    """
    historical_upath = UPath(historical_dir)
    smoothed_upath = UPath(smoothed_dir)

    smooth_config = RASTER_SMOOTH_CONFIGS.get(application_name)
    if smooth_config is None:
        raise ValueError(f"No raster smooth config for application {application_name}")

    # Check if output already exists.
    out_geojson = smoothed_upath / label / tile_fname.replace(".tif", ".geojson")
    if out_geojson.exists():
        return

    # Find historical timesteps for this tile.
    candidate_timesteps = []
    for ts_dir in historical_upath.iterdir():
        ts_label = ts_dir.name
        if ts_label > label:
            continue
        if (ts_dir / tile_fname).exists():
            candidate_timesteps.append(ts_label)
    candidate_timesteps.sort()
    timesteps = candidate_timesteps[-NUM_HISTORICAL_TIMESTEPS:]

    if not timesteps or timesteps[-1] != label:
        logger.warning(
            "Skipping tile %s: no data for label %s (found: %s)",
            tile_fname,
            label,
            timesteps,
        )
        return

    # Read CRS and transform from the first GeoTIFF.
    first_tile_path = str(historical_upath / timesteps[0] / tile_fname)
    with rasterio.open(first_tile_path) as src:
        tile_crs = src.crs
        tile_transform = src.transform

    # Load all timesteps as single-band arrays.
    images = []
    for ts in timesteps:
        tile_path = str(historical_upath / ts / tile_fname)
        with rasterio.open(tile_path) as src:
            arr = src.read(1)  # (H, W)
        images.append(torch.as_tensor(arr))
    images_tensor = torch.stack(images, dim=0)  # (T, H, W)

    num_classes = smooth_config.num_classes
    transition_probs = torch.tensor(smooth_config.transition_probs)
    emission_probs = torch.tensor(smooth_config.emission_probs)
    initial_probs = torch.tensor(smooth_config.initial_probs)

    if smooth_config.sparse:
        orig_shape = images_tensor.shape
        sel_indexes = (images_tensor.amax(dim=0) >= 2).nonzero()
        if sel_indexes.shape[0] == 0:
            logger.info("Tile %s has no non-background pixels, skipping", tile_fname)
            return
        sparse_images = images_tensor[:, sel_indexes[:, 0], sel_indexes[:, 1]]
        images_tensor = sparse_images[:, :, None]
        logger.info(
            "Extracted %d sparse indices for tile %s", sel_indexes.shape[0], tile_fname
        )

    # Viterbi forward pass.
    probs = torch.zeros(
        (images_tensor.shape[1], images_tensor.shape[2], num_classes),
        dtype=torch.float32,
    )
    probs[:, :, :] = initial_probs
    pointers = torch.zeros(
        (
            len(images_tensor),
            images_tensor.shape[1],
            images_tensor.shape[2],
            num_classes,
        ),
        dtype=torch.uint8,
    )
    for im_idx, im in enumerate(images_tensor):
        obs = (
            emission_probs.index_select(dim=1, index=im.flatten().int())
            .reshape(emission_probs.shape[0], im.shape[0], im.shape[1])
            .permute(1, 2, 0)
        )
        trans = probs[:, :, :, None] * transition_probs
        probs = trans.amax(dim=2) * obs
        pointers[im_idx, :, :, :] = trans.argmax(dim=2)

    # Viterbi backward pass.
    cur_output = probs.argmax(dim=2)
    outputs: list[torch.Tensor] = [cur_output.byte()]
    for im_idx in range(images_tensor.shape[0] - 1, 0, -1):
        cur_output = pointers[im_idx].int()[
            torch.arange(cur_output.shape[0])[:, None],
            torch.arange(cur_output.shape[1]),
            cur_output,
        ]
        outputs.append(cur_output.byte())
    outputs = list(reversed(outputs))

    if smooth_config.sparse:
        sparse_outputs = torch.stack(outputs, dim=0)[:, :, 0]
        full_outputs = torch.zeros(orig_shape, dtype=sparse_outputs.dtype)
        full_outputs[:, sel_indexes[:, 0], sel_indexes[:, 1]] = sparse_outputs
        smoothed_for_label = full_outputs[-1].numpy()
    else:
        smoothed_for_label = outputs[-1].numpy()

    # Set background (1) to 0 for simplicity.
    smoothed_for_label[smoothed_for_label == 1] = 0

    # Vectorize into polygons in the raster's native CRS, then reproject to WGS84.
    shapes_iter = rasterio.features.shapes(
        smoothed_for_label.astype(np.int32), transform=tile_transform
    )

    # Build a reprojector from the tile CRS to WGS84.
    to_wgs84 = Transformer.from_crs(tile_crs, "EPSG:4326", always_xy=True)

    features = []
    for shp, value in shapes_iter:
        if value == 0:
            continue
        geom = shapely.geometry.shape(shp)
        # Area check is in native CRS units (meters^2 for UTM).
        # MIN_AREA pixels * pixel_area gives the threshold.
        pixel_area = abs(tile_transform.a * tile_transform.e)
        if geom.area < MIN_AREA * pixel_area:
            continue
        # Simplify in CRS units (2 pixels worth of tolerance).
        geom = geom.simplify(tolerance=2 * abs(tile_transform.a))
        # Reproject to WGS84.
        geom_wgs84 = shapely.ops.transform(to_wgs84.transform, geom)
        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(geom_wgs84),
                "properties": {"value": int(value)},
            }
        )

    # Write output GeoJSON.
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    with out_geojson.open("w") as f:
        json.dump(fc, f)
    logger.info(
        "Wrote %d polygons for tile %s label %s", len(features), tile_fname, label
    )


def smooth_rasters_and_extract_polygons(
    application_name: str,
    label: str,
    historical_dir: str,
    smoothed_dir: str,
    workers: int = 16,
) -> None:
    """Smooth all raster tiles and extract polygons.

    Processes each tile independently, parallelized across workers.

    Args:
        application_name: the application name (e.g. "solar_farm").
        label: the current label (YYYY-MM).
        historical_dir: directory with per-label subdirectories of GeoTIFFs.
        smoothed_dir: directory to write smoothed polygon GeoJSONs.
        workers: number of parallel workers.
    """
    historical_upath = UPath(historical_dir)
    label_dir = historical_upath / label

    if not label_dir.exists():
        logger.warning("No predictions found at %s", label_dir)
        return

    # Enumerate tiles for the current label.
    tile_fnames = [f.name for f in label_dir.iterdir() if f.name.endswith(".tif")]
    logger.info(
        "Smoothing %d tiles for %s label %s", len(tile_fnames), application_name, label
    )

    kwargs_list = [
        dict(
            tile_fname=fname,
            historical_dir=historical_dir,
            smoothed_dir=smoothed_dir,
            label=label,
            application_name=application_name,
        )
        for fname in tile_fnames
    ]

    with multiprocessing.Pool(workers) as pool:
        outputs = star_imap_unordered(pool, _smooth_single_tile, kwargs_list)
        for _ in tqdm.tqdm(outputs, total=len(kwargs_list), desc="Smoothing tiles"):
            pass

    logger.info("Raster smoothing complete for %s label %s", application_name, label)
