"""Code to deploy forest loss driver for weekly inference run."""

import io
import json
import multiprocessing
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

from olmoearth_projects.projects.forest_loss_driver.extract_alerts import (
    ExtractAlertsArgs,
    extract_alerts,
)
from olmoearth_projects.utils.logging import get_logger

from .make_tiles import make_tiles
from .sentinel2 import get_sentinel2_assets

logger = get_logger(__name__)

BASE_URL = "https://olmoearth.allenai.org/api/v1"
ORGANIZATION_ID = "f098bcba-b994-46ce-87fc-b90b14bb8338"  # Ai2 - Demo
PROJECT_ID = (
    "2f3788b4-11bb-48ee-b379-eacccaf9734a"  # Forest Loss Driver Colombia 12 Demo
)
MODEL_ID = "03b0ae58-389f-4bd5-9bc9-01469afbfca3"  # OlmoEarth-v1-FT-ForestLossDriver-Base-20251208
# BASE_URL = "https://staging.olmoearth.allenai.org/api/v1"
# ORGANIZATION_ID = "8f51c6e0-e363-4b2d-96ca-fb9fa85e3e7e"  # Ai2 - Demo
# PROJECT_ID = (
#    "dff84383-e269-4340-81a5-4e1397934feb"  # Forest Loss Driver Colombia 12 Demo
# )
# MODEL_ID = "98c83698-e966-4075-bfc0-796ee2dd909d"  # OlmoEarth-v1-FT-ForestLossDriver-Base-20251205
REQUEST_TIMEOUT = 30
UPLOAD_TIMEOUT = 300

# Seconds to wait between polling for job status.
POLL_SLEEP_TIME = 10

# How many forest loss events to include in each Studio job.
EVENTS_PER_STUDIO_JOB = 10000


@dataclass
class IntegratedConfig:
    """Integrated inference config for forest loss driver classification.

    The arguments are combined so they can be passed together to the integrated
    pipeline, which runs the steps together in one pipeline.
    """

    # The base directory on GCS, to store outputs that should be read by the web app.
    gcs_base_dir: str
    # The base directory on WEKA, to cache intermediate outputs.
    weka_base_dir: str
    # Arguments for the extract_alerts step.
    extract_alerts_args: ExtractAlertsArgs
    # Number of workers to use to identify suitable Sentinel-2 assets for each forest
    # loss event.
    asset_workers: int
    # Number of workers to use for the make_tiles step.
    make_tiles_workers: int
    # Number of workers to use for writing individual events.
    write_individual_events_workers: int


@dataclass
class RunPaths:
    """Paths relevant to different parts of a run."""

    # Initial alerts extracted from GLAD.
    initial_alerts_fname: UPath
    # Filename to store job IDs.
    job_ids_fname: UPath
    # Outputs from Studio jobs.
    raw_studio_outputs_fname: UPath
    # Filename to write all events.
    all_events_fname: UPath
    # Directory to write per-country/month GeoJSON files.
    per_country_month_dir: UPath
    # Global (across runs) filename for latest events.
    # We use this to merge in events from the previous run.
    global_latest_fname: UPath


def get_studio_headers() -> dict[str, str]:
    """Get the headers to use for Studio API requests.

    The STUDIO_API_KEY environment variable must be set.
    """
    api_key = os.environ["STUDIO_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def _get_most_recent_friday() -> datetime:
    """Get the most recent Friday."""
    now = datetime.now()
    friday = now - timedelta(days=(now.weekday() - 4) % 7)
    return friday


def start_studio_inference_jobs(run_id: str, run_paths: RunPaths) -> list[str]:
    """Starts inference jobs on Studio.

    There is one job for each EVENTS_PER_STUDIO_JOB forest loss events. This is because
    the job seems to be stuck in pending state if there are too many events.

    If inference jobs were previously started for this run, then the previous job IDs
    are returned.

    Args:
        run_id: the run ID.
        run_paths: the paths to use for this run.

    Returns:
        the Studio job IDs.
    """
    # Read the GeoJSON, we will put it into the request.
    with run_paths.initial_alerts_fname.open() as f:
        geojson_data = json.load(f)

    # Determine the chunks of alerts, we will create one job per chunk.
    chunks = []
    for i in range(0, len(geojson_data["features"]), EVENTS_PER_STUDIO_JOB):
        chunk = geojson_data["features"][i : i + EVENTS_PER_STUDIO_JOB]
        chunks.append(chunk)
    logger.info(
        f"Got {len(chunks)} chunks with {len(geojson_data['features'])} total features"
    )

    # See if existing filename caching the job IDs exists.
    # If so, we load those already started jobs.
    if run_paths.job_ids_fname.exists():
        with run_paths.job_ids_fname.open() as f:
            job_ids_by_chunk = json.load(f)
            logger.info(
                f"Found existing job file {run_paths.job_ids_fname} with {len(job_ids_by_chunk)} jobs started already"
            )
    else:
        job_ids_by_chunk = {}

    # Start the jobs that haven't been started yet.
    for chunk_idx, chunk in enumerate(chunks):
        chunk_name = f"run_{run_id}_chunk_{chunk_idx}"
        if chunk_name in job_ids_by_chunk:
            continue

        # API request to run the model.
        logger.info(
            f"Starting a prediction job with {len(chunk)} features named {chunk_name}"
        )
        json_request_data = {
            "fine_tuned_model_id": MODEL_ID,
            "name": chunk_name,
            "project_id": PROJECT_ID,
            "geojson": {
                "type": "FeatureCollection",
                "properties": {},
                "features": chunk,
            },
        }
        url = f"{BASE_URL}/predictions"
        response = requests.post(
            url,
            json=json_request_data,
            timeout=(REQUEST_TIMEOUT, UPLOAD_TIMEOUT),
            headers=get_studio_headers(),
        )
        response.raise_for_status()

        json_data = response.json()
        if "records" not in json_data or len(json_data["records"]) != 1:
            raise ValueError(
                f"expected response to have one record, but got {json_data}"
            )

        job_ids_by_chunk[chunk_name] = json_data["records"][0]["id"]

        with open_atomic(run_paths.job_ids_fname, "w") as f:
            json.dump(job_ids_by_chunk, f)

    return list(job_ids_by_chunk.values())


def wait_for_studio_job(job_id: str, max_consecutive_errors: int = 3) -> None:
    """Wait for Studio prediction job to finish successfully.

    Raises an exception if the job fails.

    Args:
        job_id: the job ID to check.
        max_consecutive_errors: maximum consecutive connection/timeout/response-format
            errors before giving up.
    """

    def check_job_status() -> str:
        url = f"{BASE_URL}/predictions/{job_id}"
        response = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers=get_studio_headers()
        )
        response.raise_for_status()

        json_data = response.json()
        if "records" not in json_data or len(json_data["records"]) != 1:
            raise ValueError(
                f"expected response to have one record, but got {json_data}"
            )

        return json_data["records"][0]["status"]

    consecutive_errors = 0
    while True:
        try:
            job_status = check_job_status()
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors > max_consecutive_errors:
                raise
            logger.warning(f"Got error while polling job status, trying again: {e}")
            time.sleep(POLL_SLEEP_TIME)
            continue

        logger.debug(f"Polled job status, status is {job_status}")
        if job_status in ["pending", "predicting"]:
            time.sleep(POLL_SLEEP_TIME)
            continue
        elif (
            job_status not in ["completed", "failed"]
        ):  # tmp: we need to treat failed as completed since some jobs are stuck failed but have outputs
            raise ValueError(
                f"expected status to be pending or completed, but got {job_status}"
            )

        break


def get_prediction_result(job_id: str) -> list[Feature]:
    """Get the FeatureCollection result from a Studio job.

    Args:
        job_id: the Studio Prediction job ID.

    Returns:
        the FeatureCollection dict.
    """
    # Get the prediction results download_token.
    url = f"{BASE_URL}/predictions/{job_id}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=get_studio_headers())
    response.raise_for_status()
    json_data = response.json()
    if "records" not in json_data or len(json_data["records"]) != 1:
        raise ValueError(f"expected response to have one record, but got {json_data}")
    download_token = json_data["records"][0]["result"]["download_token"]

    # Save the prediction result GeoJSON to a temp file.
    url = f"{BASE_URL}/prediction-results/files?download_token={download_token}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_fname = UPath(tmp_dir) / "data.geojson"

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The download is zip archive which for our jobs should contain a single
            # GeoJSON file. ".geojson" may appear in the middle of the name though, in
            # case filename is like result.geojson?Expires=...&Signature=...
            fnames = z.namelist()
            if len(fnames) != 1 or ".geojson" not in fnames[0]:
                raise ValueError(
                    f"expected prediction result zip file to contain one GeoJSON file but got {fnames}"
                )

            with z.open(fnames[0]) as src, tmp_fname.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        # Read the features.
        return GeojsonVectorFormat().decode_from_file(tmp_fname)


def get_prediction_results(job_ids: list[str], run_paths: RunPaths) -> list[Feature]:
    """Get and cache prediction results across many Studio jobs.

    Args:
        job_ids: list of Studio job IDs to get results for.
        run_paths: paths to use for this run.

    Returns:
        list of output features concatenated across jobs.
    """
    # See if this operation has been completed already.
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    if run_paths.raw_studio_outputs_fname.exists():
        logger.info(
            f"Loading previously downloaded Studio job outputs from {run_paths.raw_studio_outputs_fname}"
        )
        return vector_format.decode_from_file(run_paths.raw_studio_outputs_fname)

    forest_loss_events: list[Feature] = []
    for job_id in job_ids:
        logger.info(f"Getting forest loss event outputs from job {job_id}")
        forest_loss_events.extend(get_prediction_result(job_id))

    # Cache and return the events.
    vector_format.encode_to_file(run_paths.raw_studio_outputs_fname, forest_loss_events)
    return forest_loss_events


def add_input_properties_to_output_features(
    input_features: list[Feature], output_features: list[Feature]
) -> None:
    """Add properties from the corresponding input feature to each output feature.

    The input features should be a superset of the output features, but features could
    have slight changes due to floating point rounding. We assume that the input
    feature with the highest intersection area to the output feature is the matching
    one. We raise error if this area is less than half of the feature's area.
    """
    # 0.01 should give a reasonable number of grid cells (100 pixels).
    grid_index = GridIndex(0.01)

    # Insert input features into the grid index.
    for input_feat in input_features:
        wgs84_geom = input_feat.geometry.to_projection(WGS84_PROJECTION)
        grid_index.insert(
            wgs84_geom.shp.bounds, (wgs84_geom.shp, input_feat.properties)
        )

    # Get closest input feature to each output feature.
    # And add the input properties.
    for output_feat in output_features:
        output_wgs84_geom = output_feat.geometry.to_projection(WGS84_PROJECTION)
        candidates: list[tuple[shapely.Geometry, dict]] = grid_index.query(
            output_wgs84_geom.shp.bounds
        )
        best_candidate_props: dict | None = None
        best_candidate_score: float | None = None
        for input_wgs84_shp, input_props in candidates:
            score = input_wgs84_shp.intersection(output_wgs84_geom.shp).area
            if score < output_wgs84_geom.shp.area / 2:
                continue
            if best_candidate_score is None or score > best_candidate_score:
                best_candidate_props = input_props
                best_candidate_score = score

        if best_candidate_props is None:
            raise ValueError(f"found no input feature for output feature {output_feat}")

        output_feat.properties.update(best_candidate_props)


def merge_forest_loss_events(
    inference_job_ids: list[str], asset_workers: int, run_paths: RunPaths
) -> list[Feature]:
    """Get the forest loss events from Studio and merge them with previous events.

    We also determine Sentinel-2 assets to display for the new events, and add an index
    property to all events.

    The whole operation is cached, so if it has already been completed we don't repeat
    it.

    Args:
        inference_job_ids: the Studio job IDs for this run.
        asset_workers: number of workers for getting Sentinel-2 assets.
        run_paths: paths to use for this run.

    Returns:
        list of merged forest loss events.
    """
    # See if this operation has been completed already.
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    if run_paths.all_events_fname.exists():
        logger.info(
            f"Loading previously computed merged events from {run_paths.all_events_fname}"
        )
        return vector_format.decode_from_file(run_paths.all_events_fname)

    # Get prediction result from Studio.
    forest_loss_events = get_prediction_results(inference_job_ids, run_paths)

    # Add back properties we had on our original features, like "country".
    input_features = vector_format.decode_from_file(run_paths.initial_alerts_fname)
    add_input_properties_to_output_features(input_features, forest_loss_events)

    # Rename new_label to category in the feature properties.
    # Also delete some excess properties.
    for event in forest_loss_events:
        event.properties["category"] = event.properties["new_label"]
        del event.properties["new_label"]
        for prop_name in [
            "oe_prediction_result_id",
            "oe_prediction_result_file_id",
            "oe_created_at",
        ]:
            if prop_name not in event.properties:
                continue
            del event.properties[prop_name]

    # Identify which Planetary Computer asset to read for each forest loss event.
    get_sentinel2_assets(forest_loss_events, workers=asset_workers)

    # Get the earliest oe_start_time across all of the new forest loss events.
    # We use this during merging to avoid adding anything with an equal or higher start
    # time.
    earliest_start_time: str | None = None
    for event in forest_loss_events:
        if (
            earliest_start_time is None
            or event.properties["oe_start_time"] < earliest_start_time
        ):
            earliest_start_time = event.properties["oe_start_time"]

    # Merge in pre-existing forest loss events (if any).
    if run_paths.global_latest_fname.exists():
        previous_events = vector_format.decode_from_file(run_paths.global_latest_fname)
        logger.info(
            f"Merging in a subset of the {len(previous_events)} previously computed events from {run_paths.global_latest_fname}, currently this run has {len(forest_loss_events)} events"
        )

        for event in previous_events:
            if event.properties["oe_start_time"] >= earliest_start_time:
                continue
            forest_loss_events.append(event)

        logger.info(f"After merging, this run has {len(forest_loss_events)} events")
    else:
        logger.info(
            f"Skipping merging since {run_paths.global_latest_fname} does not exist"
        )

    # Assign index property indicating index in the list.
    for index, event in enumerate(forest_loss_events):
        event.properties["index"] = index

    # Cache and return the events.
    vector_format.encode_to_file(run_paths.all_events_fname, forest_loss_events)
    return forest_loss_events


def write_individual_event(fname: UPath, event: Feature) -> None:
    """Write an event to the specified file."""
    GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84).encode_to_file(
        fname, [event]
    )


def write_individual_events(
    dst_dir: UPath, events: list[Feature], num_workers: int
) -> None:
    """Write the events to individual files in dst_dir.

    This way the web app can reach an event separately when the user requests to view
    its details.

    Args:
        dst_dir: the directory to write the per-event files.
        events: list of events to write.
        num_workers: number of worker processes to use.
    """
    p = multiprocessing.Pool(num_workers)
    write_individual_event_args = [
        dict(
            fname=dst_dir / f"feat_{event.properties['index']}",
            event=event,
        )
        for event in events
    ]
    outputs = star_imap_unordered(
        p, write_individual_event, write_individual_event_args
    )
    for _ in tqdm.tqdm(
        outputs,
        total=len(write_individual_event_args),
        desc="Writing individual event files",
    ):
        pass
    p.close()


def integrated_pipeline(integrated_config: IntegratedConfig) -> None:
    """Integrated pipeline that runs all stages for forest loss driver model.

    1. Process GLAD alerts to get prediction request geometry for recent forest loss
       events.
    2. Make OlmoEarth API request to run inference.
    3. Merge the new events with existing ones.
    4. Get metadata for Planetary Computer scenes that can be used to visualize the
       before and after images for each forest loss event.
    5. Use tippecanoe to make tiles, and upload those tiles along with GeoJSON files to
       GCS so the website can access it.

    Args:
        integrated_config: the integrated configuration for all inference pipeline
            steps.
    """
    # Make run ID based on the current time.
    # We check the most recent Friday since that is when GLAD alerts are released, and
    # this way we will have the same run ID in case of restarts.
    run_id = _get_most_recent_friday().strftime("%Y%m%d")
    weka_ds_root = UPath(integrated_config.weka_base_dir) / f"dataset_{run_id}"
    gcs_ds_root = UPath(integrated_config.gcs_base_dir) / f"dataset_{run_id}"

    run_paths = RunPaths(
        initial_alerts_fname=weka_ds_root / "prediction_request_geometry.geojson",
        job_ids_fname=weka_ds_root / "job_id.json",
        raw_studio_outputs_fname=weka_ds_root / "events_from_studio_jobs.geojson",
        all_events_fname=gcs_ds_root / "all_events.geojson",
        per_country_month_dir=gcs_ds_root,
        global_latest_fname=gcs_ds_root.parent / "latest.geojson",
    )

    integrated_config.extract_alerts_args.out_fname = str(
        run_paths.initial_alerts_fname
    )

    # Create prediction request geometry.
    if not run_paths.initial_alerts_fname.exists():
        logger.info(
            f"Computing prediction request geometry at {run_paths.initial_alerts_fname}"
        )
        extract_alerts(integrated_config.extract_alerts_args)
    else:
        logger.info(
            f"Using existing prediction request geometry at {run_paths.initial_alerts_fname}"
        )

    # Start the Studio inference job.
    inference_job_ids = start_studio_inference_jobs(run_id, run_paths)
    logger.info(f"Got Studio inference job IDs: {inference_job_ids}")

    # Check job status.
    for job_id in inference_job_ids:
        wait_for_studio_job(job_id)

    # Get forest loss events from Studio, identify Sentinel-2 assets for visualization
    # for each event, and merge in previous events before the time window we are
    # processing.
    # This function will also save all_events.geojson in gcs_ds_root.
    forest_loss_events = merge_forest_loss_events(
        inference_job_ids=inference_job_ids,
        asset_workers=integrated_config.asset_workers,
        run_paths=run_paths,
    )

    # Write latest.geojson (used for merging) and per-country/month files.
    # We write latest.geojson here instead of in merge_forest_loss_events since it is a
    # cleaner way to ensure we always have completed this step (we don't want to write
    # all_events_fname and then job crashes and we don't write latest_events_fname).
    logger.info(
        f"Got {len(forest_loss_events)} after merging, writing latest.geojson and per-country/month files to GCS"
    )
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    vector_format.encode_to_file(run_paths.global_latest_fname, forest_loss_events)

    events_by_country_month: dict[tuple[str, str], list[Feature]] = {}
    for event in forest_loss_events:
        country = event.properties.get("country", "unknown")
        year_and_month = event.properties["oe_start_time"][0:7]
        k = (country, year_and_month)
        if k not in events_by_country_month:
            events_by_country_month[k] = []
        events_by_country_month[k].append(event)

    for (country, year_and_month), cur_events in events_by_country_month.items():
        out_fname = (
            run_paths.per_country_month_dir / f"{country}_{year_and_month}.geojson"
        )
        vector_format.encode_to_file(out_fname, cur_events)

    # Make slippy tiles.
    logger.info("Making slippy tiles")
    make_tiles(
        workers=integrated_config.make_tiles_workers,
        in_fname=str(run_paths.all_events_fname),
        gcs_ds_root=str(gcs_ds_root),
    )

    # Write each individual event.
    write_individual_events(
        dst_dir=gcs_ds_root / "events",
        events=forest_loss_events,
        num_workers=integrated_config.write_individual_events_workers,
    )

    # Write metadata file about the latest run.
    # The web app will monitor for these files to determine when to reload.
    logger.info("Writing run_metadata.json")
    run_metadata = {
        "run_id": run_id,
        "country_months": list(events_by_country_month.keys()),
        "num_events": len(forest_loss_events),
    }
    with (gcs_ds_root / "run_metadata.json").open("w") as f:
        json.dump(run_metadata, f)


workflows = {
    "integrated_pipeline": integrated_pipeline,
}
