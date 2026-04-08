"""Studio API client for submitting and monitoring prediction jobs."""

import json
import os
import random
import time
from datetime import datetime

import requests
from upath import UPath

from olmoearth_projects.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://olmoearth.allenai.org/api/v1"
REQUEST_TIMEOUT = 30
UPLOAD_TIMEOUT = 300
POLL_SLEEP_TIME = 30


def _get_headers() -> dict[str, str]:
    """Get headers for Studio API requests."""
    api_key = os.environ["STUDIO_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def _start_prediction_job(
    model_id: str,
    project_id: str,
    tile: dict,
    time_range: tuple[datetime, datetime],
) -> str:
    """Submit a single prediction job to the Studio API.

    Args:
        model_id: the fine-tuned model ID.
        project_id: the Studio project ID.
        tile: GeoJSON Feature dict for the tile.
        time_range: (start, end) time range for prediction.

    Returns:
        the prediction job ID.
    """
    tile_name = tile["properties"]["tile_name"]

    time_props = {
        "oe_start_time": time_range[0].isoformat(),
        "oe_end_time": time_range[1].isoformat(),
    }

    # Use 1x1-degree sub-features when available; fall back to the whole tile
    # for backwards compatibility with old cached tiles.
    raw_features = tile.get("features") or [tile]
    features = []
    for f in raw_features:
        feat = dict(f)
        feat["properties"] = {**feat.get("properties", {}), **time_props}
        features.append(feat)

    json_request_data = {
        "model_id": model_id,
        "name": tile_name,
        "project_id": project_id,
        "geojson": {
            "type": "FeatureCollection",
            "properties": {},
            "features": features,
        },
        # We expect that some tiles will have many failing windows due to no Sentinel-2
        # coverage in part or all of the tile. So we set the required success rate to 0.
        "min_window_success_rate": 0,
    }
    url = f"{BASE_URL}/predictions"
    response = requests.post(
        url,
        json=json_request_data,
        timeout=(REQUEST_TIMEOUT, UPLOAD_TIMEOUT),
        headers=_get_headers(),
    )
    response.raise_for_status()

    json_data = response.json()
    if "records" not in json_data or len(json_data["records"]) != 1:
        raise ValueError(f"expected response to have one record, but got {json_data}")

    return json_data["records"][0]["id"]


def _check_job_status(job_id: str) -> str:
    """Check the status of a Studio prediction job.

    Args:
        job_id: the job ID to check.

    Returns:
        the job status string.
    """
    url = f"{BASE_URL}/predictions/{job_id}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=_get_headers())
    response.raise_for_status()
    json_data = response.json()
    if "records" not in json_data or len(json_data["records"]) != 1:
        raise ValueError(f"expected response to have one record, but got {json_data}")
    return json_data["records"][0]["status"]


def _download_prediction_result(job_id: str, dst_path: UPath) -> None:
    """Download prediction results from a completed Studio job.

    Args:
        job_id: the completed job ID.
        dst_path: path to save the result file.
    """
    url = f"{BASE_URL}/predictions/{job_id}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=_get_headers())
    response.raise_for_status()
    json_data = response.json()
    download_token = json_data["records"][0]["result"]["download_token"]

    url = f"{BASE_URL}/prediction-results/files?download_token={download_token}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("wb") as f:
        f.write(response.content)


def submit_and_wait_for_jobs(
    tiles: list[dict],
    model_id: str,
    project_id: str,
    time_range: tuple[datetime, datetime],
    job_ids_path: UPath,
    results_dir: UPath,
    max_concurrent: int = 20,
) -> None:
    """Submit prediction jobs for all tiles, wait for them, and download results.

    Manages concurrency by keeping at most max_concurrent jobs running at
    once. Caches job IDs to job_ids_path to support restarts. Downloaded
    results are saved to results_dir as {tile_name}.zip.

    Args:
        tiles: list of tile GeoJSON Feature dicts.
        model_id: Studio model ID.
        project_id: Studio project ID.
        time_range: prediction time range.
        job_ids_path: path to cache job ID mapping.
        results_dir: directory to save downloaded results.
        max_concurrent: max simultaneous running jobs.
    """
    # Load or initialize job ID cache: tile_name -> job_id.
    if job_ids_path.exists():
        with job_ids_path.open() as f:
            job_ids: dict[str, str] = json.load(f)
        logger.info("Loaded %d cached job IDs from %s", len(job_ids), job_ids_path)
    else:
        job_ids = {}

    # Track which tiles already have results downloaded.
    downloaded = set()
    if results_dir.exists():
        for p in results_dir.iterdir():
            downloaded.add(p.name)

    tile_by_name = {t["properties"]["tile_name"]: t for t in tiles}

    # Determine tiles still needing work.
    pending_tile_names = []
    for tile in tiles:
        name = tile["properties"]["tile_name"]
        if f"{name}.zip" in downloaded:
            continue
        pending_tile_names.append(name)

    logger.info(
        "%d tiles total, %d already downloaded, %d pending",
        len(tiles),
        len(downloaded),
        len(pending_tile_names),
    )

    submitted_names = [n for n in pending_tile_names if n in job_ids]
    not_submitted_names = [n for n in pending_tile_names if n not in job_ids]

    # Make sure we submit jobs in random order.
    random.shuffle(not_submitted_names)

    active: dict[str, str] = {}  # tile_name -> job_id
    for name in submitted_names:
        active[name] = job_ids[name]

    submit_idx = 0

    def _save_job_ids() -> None:
        job_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with job_ids_path.open("w") as f:
            json.dump(job_ids, f)

    while active or submit_idx < len(not_submitted_names):
        # Submit new jobs up to concurrency limit.
        while len(active) < max_concurrent and submit_idx < len(not_submitted_names):
            name = not_submitted_names[submit_idx]
            submit_idx += 1
            tile = tile_by_name[name]
            try:
                job_id = _start_prediction_job(model_id, project_id, tile, time_range)
                job_ids[name] = job_id
                active[name] = job_id
                _save_job_ids()
                logger.info("Submitted job for tile %s -> %s", name, job_id)
            except Exception:
                logger.exception("Failed to submit job for tile %s", name)
                raise

        if not active:
            break

        # Poll active jobs.
        time.sleep(POLL_SLEEP_TIME)
        completed_names = []
        for name, job_id in list(active.items()):
            try:
                status = _check_job_status(job_id)
            except Exception:
                logger.warning(
                    "Error polling job %s for tile %s, will retry", job_id, name
                )
                continue

            if status in ("pending", "predicting"):
                continue
            elif status in ("completed", "failed"):
                dst = results_dir / f"{name}.zip"
                try:
                    _download_prediction_result(job_id, dst)
                    logger.info(
                        "Downloaded result for tile %s (status=%s)", name, status
                    )
                except Exception:
                    logger.warning(
                        "Failed to download result for tile %s job %s",
                        name,
                        job_id,
                    )
                    continue
                completed_names.append(name)
            else:
                logger.warning(
                    "Unexpected job status %s for tile %s job %s",
                    status,
                    name,
                    job_id,
                )

        for name in completed_names:
            del active[name]

        remaining = len(active) + (len(not_submitted_names) - submit_idx)
        logger.info(
            "Progress: %d active, %d remaining to submit, %d total remaining",
            len(active),
            len(not_submitted_names) - submit_idx,
            remaining,
        )

    logger.info("All %d tiles completed", len(tiles))
