"""Publish Satlas smoothed outputs to a public UPath.

Writes smoothed GeoJSONs and maintains an index.json listing all available
files with their metadata.
"""

import json
from datetime import UTC, datetime

from upath import UPath

from olmoearth_projects.utils.logging import get_logger

logger = get_logger(__name__)

# Number of most recent timesteps to publish for point applications.
NUM_RECENT_TIMESTEPS = 6


def _update_index(publish_upath: UPath) -> None:
    """Update index.json at the given path.

    Lists all files (excluding index.json itself) with their names and
    the current timestamp.

    Args:
        publish_upath: the publish directory.
    """
    entries = []
    for fpath in publish_upath.iterdir():
        if fpath.name == "index.json":
            continue
        entries.append(
            {
                "name": fpath.name,
                "updated_at": datetime.now(tz=UTC).isoformat(),
            }
        )
    entries.sort(key=lambda e: e["name"])

    index = {"files": entries}
    index_path = publish_upath / "index.json"
    with index_path.open("w") as f:
        json.dump(index, f, indent=2)
    logger.info("Updated index.json with %d entries at %s", len(entries), index_path)


def publish_points(
    smoothed_path: str,
    publish_path: str,
) -> None:
    """Publish smoothed point outputs.

    Copies the most recent timestep GeoJSONs, latest.geojson, and
    history.geojson to the publish path, then updates index.json.

    Args:
        smoothed_path: directory containing smoothed per-timestep GeoJSONs and
            history.geojson.
        publish_path: public UPath to publish to.
    """
    smoothed_upath = UPath(smoothed_path)
    publish_upath = UPath(publish_path)
    publish_upath.mkdir(parents=True, exist_ok=True)

    # Collect per-timestep files (exclude history.geojson).
    timestep_files: list[UPath] = []
    for fpath in smoothed_upath.iterdir():
        if fpath.name == "history.geojson":
            continue
        if not fpath.name.endswith(".geojson"):
            continue
        timestep_files.append(fpath)
    timestep_files.sort(key=lambda p: p.name)

    # Copy the most recent N timesteps.
    for fpath in timestep_files[-NUM_RECENT_TIMESTEPS:]:
        dst = publish_upath / fpath.name
        logger.info("Publishing %s -> %s", fpath, dst)
        _copy_file(fpath, dst)

    # Copy the most recent as latest.geojson.
    if timestep_files:
        latest_src = timestep_files[-1]
        latest_dst = publish_upath / "latest.geojson"
        logger.info("Publishing latest.geojson from %s", latest_src.name)
        _copy_file(latest_src, latest_dst)

    # Copy history.geojson.
    history_src = smoothed_upath / "history.geojson"
    if history_src.exists():
        history_dst = publish_upath / "history.geojson"
        logger.info("Publishing history.geojson")
        _copy_file(history_src, history_dst)

    _update_index(publish_upath)
    logger.info("Point publishing complete to %s", publish_path)


def publish_rasters(
    smoothed_path: str,
    publish_path: str,
    label: str,
) -> None:
    """Publish smoothed raster (vectorized polygon) outputs.

    Copies all polygon GeoJSONs for the given label to the publish path,
    then updates index.json.

    Args:
        smoothed_path: directory containing per-label subdirectories with
            vectorized polygon GeoJSONs.
        publish_path: public UPath to publish to.
        label: the label to publish.
    """
    smoothed_upath = UPath(smoothed_path)
    publish_upath = UPath(publish_path) / label
    publish_upath.mkdir(parents=True, exist_ok=True)

    label_dir = smoothed_upath / label
    if not label_dir.exists():
        logger.warning("No smoothed raster outputs at %s", label_dir)
        return

    count = 0
    for fpath in label_dir.iterdir():
        if not fpath.name.endswith(".geojson"):
            continue
        dst = publish_upath / fpath.name
        _copy_file(fpath, dst)
        count += 1

    _update_index(publish_upath)
    # Also update the parent index to list all labels.
    _update_index(publish_upath.parent)

    logger.info(
        "Raster publishing complete: %d files for label %s to %s",
        count,
        label,
        publish_path,
    )


def _copy_file(src: UPath, dst: UPath) -> None:
    """Copy a file from src to dst, supporting cloud paths."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fsrc:
        with dst.open("wb") as fdst:
            while True:
                chunk = fsrc.read(8 * 1024 * 1024)
                if not chunk:
                    break
                fdst.write(chunk)
