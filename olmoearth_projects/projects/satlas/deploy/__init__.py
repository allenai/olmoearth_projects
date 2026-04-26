"""Satlas deploy pipeline: orchestrate prediction, smoothing, and publishing."""

import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from global_land_mask import globe
from upath import UPath

from olmoearth_projects.utils.logging import get_logger

from ..publish import publish_points, publish_rasters
from ..smooth_points import smooth_points
from ..smooth_rasters import smooth_rasters_and_extract_polygons
from ..studio_api import submit_and_wait_for_jobs

logger = get_logger(__name__)


class Application(Enum):
    """Satlas applications."""

    MARINE_INFRA = "marine_infra"
    WIND_TURBINE = "wind_turbine"
    SOLAR_FARM = "solar_farm"


# Whether the application produces raster (segmentation) vs point (detection) output.
APP_IS_RASTER = {
    Application.MARINE_INFRA: False,
    Application.WIND_TURBINE: False,
    Application.SOLAR_FARM: True,
}


@dataclass
class IntegratedConfig:
    """Configuration for the Satlas integrated deploy pipeline."""

    # Which application to run.
    application: Application
    # Studio API model ID for this application.
    model_id: str
    # Studio API project ID.
    project_id: str
    # Size of each tile in degrees.
    tile_size_degrees: int = 5
    # Maximum number of concurrent Studio API jobs.
    max_concurrent_jobs: int = 20
    # Path to publish final smoothed outputs.
    publish_path: str = ""
    # Base directory on WEKA for intermediate outputs.
    weka_base_dir: str = ""
    # Optional label override (YYYY-MM). Defaults to most recent completed quarter.
    label: str = ""
    # Number of workers for smoothing parallelization.
    smooth_workers: int = 16


def get_default_label() -> str:
    """Compute the default label (most recent completed quarter).

    Returns the YYYY-MM of the most recent quarter whose 3-month window has
    fully elapsed. E.g. on 2026-02-13, Q1 2026 (Jan-Mar) is still running,
    so we return 2025-10 (Q4 2025, which ended 2026-01-01).
    """
    now = datetime.now(tz=UTC)
    year = now.year
    month = now.month
    quarter_start_months = [1, 4, 7, 10]
    current_quarter_month = max(m for m in quarter_start_months if m <= month)
    idx = quarter_start_months.index(current_quarter_month)
    if idx == 0:
        year -= 1
        prev_quarter_month = quarter_start_months[-1]
    else:
        prev_quarter_month = quarter_start_months[idx - 1]
    return f"{year:04d}-{prev_quarter_month:02d}"


def label_to_time_range(label: str) -> tuple[datetime, datetime]:
    """Convert a YYYY-MM label to a [start, end) time range.

    The time range covers 90 days before and 90 days after the label motnh.

    Args:
        label: YYYY-MM format label.

    Returns:
        (start_datetime, end_datetime) tuple with timezone-aware datetimes.
    """
    year, month = int(label[:4]), int(label[5:7])
    label_ts = datetime(year, month, 1, tzinfo=UTC)
    return (label_ts - timedelta(days=90), label_ts + timedelta(days=90))


def generate_tiles(
    tile_size_degrees: int,
    application: Application,
) -> list[dict]:
    """Generate NxN degree tiles covering the world with land/ocean filtering.

    For each tile, samples a grid of points at every 1 degree and checks
    land vs ocean. Skips tiles that are entirely the wrong surface type:
    - marine_infra: skip if all sample points are on land
    - wind_turbine/solar_farm: skip if all sample points are on ocean

    Args:
        tile_size_degrees: size of each tile in degrees.
        application: which application (controls land/ocean filter).

    Returns:
        list of GeoJSON Feature dicts, each with a Polygon geometry for the
        tile bounds and properties including tile_name.
    """
    n = tile_size_degrees
    tiles = []
    wants_ocean = application == Application.MARINE_INFRA

    for lon_base in range(-180, 180, n):
        for lat_base in range(-70, 70, n):
            # Clip longitude to [-179, 179] so we don't have antimeridian issues.
            lon_start = max(lon_base, -179)
            lat_start = max(lat_base, -70)
            lon_end = min(lon_start + n, 179)
            lat_end = min(lat_start + n, 70)

            # Build 1x1-degree sub-features, keeping only cells that pass
            # the land/ocean filter.
            sub_features = []
            for cell_lon in range(lon_start, lon_end):
                for cell_lat in range(lat_start, lat_end):
                    cell_lon_end = cell_lon + 1
                    cell_lat_end = cell_lat + 1

                    has_land = has_ocean = False
                    for clon in [cell_lon, cell_lon_end]:
                        for clat in [cell_lat, cell_lat_end]:
                            if globe.is_land(clat, clon):
                                has_land = True
                            else:
                                has_ocean = True

                    if wants_ocean and not has_ocean:
                        continue
                    if not wants_ocean and not has_land:
                        continue

                    sub_features.append(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [cell_lon, cell_lat],
                                        [cell_lon_end, cell_lat],
                                        [cell_lon_end, cell_lat_end],
                                        [cell_lon, cell_lat_end],
                                        [cell_lon, cell_lat],
                                    ]
                                ],
                            },
                            "properties": {},
                        }
                    )

            if not sub_features:
                continue

            tile_name = f"{lon_base}_{lat_base}"
            tile = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [lon_start, lat_start],
                            [lon_end, lat_start],
                            [lon_end, lat_end],
                            [lon_start, lat_end],
                            [lon_start, lat_start],
                        ]
                    ],
                },
                "properties": {
                    "tile_name": tile_name,
                },
                "features": sub_features,
            }
            tiles.append(tile)

    logger.info(
        "Generated %d tiles (tile_size=%d, application=%s)",
        len(tiles),
        n,
        application.value,
    )
    return tiles


def integrated_pipeline(integrated_config: IntegratedConfig) -> None:
    """Run the full Satlas prediction pipeline for one application.

    Steps:
    1. Compute label and time range.
    2. Generate tiles with land/ocean filtering.
    3. Submit prediction jobs via Studio API with concurrency limiting.
    4. Wait for all jobs to complete and download results.
    5. Apply temporal smoothing (point Viterbi or raster Viterbi + vectorize).
    6. Publish smoothed outputs.

    Args:
        integrated_config: pipeline configuration.
    """
    cfg = integrated_config
    application = cfg.application
    is_raster = APP_IS_RASTER[application]

    # 1. Compute label and time range.
    label = cfg.label if cfg.label else get_default_label()
    time_range = label_to_time_range(label)
    logger.info(
        "Running %s for label=%s, time_range=%s to %s",
        application.value,
        label,
        time_range[0].isoformat(),
        time_range[1].isoformat(),
    )

    # Set up paths.
    weka_dir = UPath(cfg.weka_base_dir) / application.value / label
    historical_dir = UPath(cfg.weka_base_dir) / application.value / "history"
    results_dir = historical_dir / label
    job_ids_path = weka_dir / "job_ids.json"

    # 2. Generate tiles (or load from cache).
    tiles_cache_path = weka_dir / "tiles.json"
    if tiles_cache_path.exists():
        logger.info("Loading tiles from cache: %s", tiles_cache_path)
        with tiles_cache_path.open() as f:
            tiles = json.load(f)
    else:
        tiles = generate_tiles(cfg.tile_size_degrees, application)
        tiles_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tiles_cache_path.open("w") as f:
            json.dump(tiles, f)
        logger.info("Saved tiles cache to %s", tiles_cache_path)

    # 3-4. Submit jobs and wait for completion (downloads .zip per tile).
    zip_dir = weka_dir / "zips"
    submit_and_wait_for_jobs(
        tiles=tiles,
        model_id=cfg.model_id,
        project_id=cfg.project_id,
        time_range=time_range,
        job_ids_path=job_ids_path,
        results_dir=zip_dir,
        max_concurrent=cfg.max_concurrent_jobs,
    )

    # 4b. Unzip results: each .zip contains a single GeoTIFF (for raster) or
    # GeoJSON (for points). Extract into the historical results_dir so that
    # the smoothing step can find them.
    results_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_dir.iterdir():
        if not zip_path.name.endswith(".zip"):
            continue
        tile_name = zip_path.name.removesuffix(".zip")
        with zipfile.ZipFile(str(zip_path)) as zf:
            for member in zf.namelist():
                ext = member.rsplit(".", 1)[-1] if "." in member else ""
                dst = results_dir / f"{tile_name}.{ext}"
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dst.open("wb") as out:
                    out.write(src.read())
        logger.info("Extracted %s -> %s", zip_path.name, results_dir)

    # 4c. For point applications, merge per-tile GeoJSONs into a single
    # LABEL.geojson in historical_dir, as expected by smooth_points.
    if not is_raster:
        merged_geojson_path = historical_dir / f"{label}.geojson"
        if merged_geojson_path.exists():
            logger.info(
                "Merged GeoJSON already exists at %s, skipping merge step",
                merged_geojson_path,
            )
        else:
            all_features = []
            for tile_path in results_dir.iterdir():
                if not tile_path.name.endswith(".geojson"):
                    continue
                with tile_path.open() as f:
                    fc = json.load(f)
                all_features.extend(fc.get("features", []))
            merged_fc = {"type": "FeatureCollection", "features": all_features}
            historical_dir.mkdir(parents=True, exist_ok=True)
            with merged_geojson_path.open("w") as f:
                json.dump(merged_fc, f)
            logger.info(
                "Merged %d features from %d tiles into %s",
                len(all_features),
                sum(1 for p in results_dir.iterdir() if p.name.endswith(".geojson")),
                merged_geojson_path,
            )

    # 5. Apply temporal smoothing.
    smoothed_dir = UPath(cfg.weka_base_dir) / application.value / "smoothed"

    if is_raster:
        logger.info("Running raster smoothing and polygon extraction")
        smooth_rasters_and_extract_polygons(
            application_name=application.value,
            label=label,
            historical_dir=str(historical_dir),
            smoothed_dir=str(smoothed_dir),
            workers=cfg.smooth_workers,
        )
    else:
        logger.info("Running point smoothing")
        smooth_points(
            label=label,
            historical_dir=str(historical_dir),
            smoothed_dir=str(smoothed_dir),
        )

    # 6. Publish.
    publish_dir = UPath(cfg.publish_path) / application.value
    if is_raster:
        publish_rasters(
            smoothed_path=str(smoothed_dir),
            publish_path=str(publish_dir),
            label=label,
        )
    else:
        publish_points(
            smoothed_path=str(smoothed_dir),
            publish_path=str(publish_dir),
        )

    logger.info("Pipeline complete for %s label=%s", application.value, label)


workflows = {
    "integrated_pipeline": integrated_pipeline,
}
