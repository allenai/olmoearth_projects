"""Code to deploy forest loss driver for weekly inference run."""

import copy
import io
import json
import multiprocessing
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import requests
import shapely
import shapely.geometry
import tqdm
from rslearn.utils.feature import Feature
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

from olmoearth_projects.projects.forest_loss_driver.extract_alerts import (
    ExtractAlertsArgs,
    extract_alerts,
)
from olmoearth_projects.utils.logging import get_logger
from olmoearth_projects.utils.studio_client import StudioClient

from .centroid_index import CentroidIndex
from .make_tiles import make_tiles
from .monocrop import (
    add_monoculture_predictions,
    make_monocrop_request_features,
    select_monocrop_events,
)
from .overlap_dedup import filter_overlapping_previous_events
from .sentinel2 import get_sentinel2_assets

logger = get_logger(__name__)

ORGANIZATION_ID = "f098bcba-b994-46ce-87fc-b90b14bb8338"  # Ai2 - Demo
PROJECT_ID = (
    "2f3788b4-11bb-48ee-b379-eacccaf9734a"  # Forest Loss Driver Colombia 12 Demo
)
# The forest loss driver model, which classifies the driver of each forest loss event.
MODEL_ID = "a3c3e819-7aa9-47e9-98fa-f72449a56263"
# The monoculture model (see olmoearth_run_data/forest_loss_driver_monocrop/), which
# classifies the type of agriculture for large-scale agriculture events.
MONOCROP_MODEL_ID = "522c3aa3-77aa-49d3-a215-a3bfed8dac55"

# Timeout (seconds) for downloading prediction results.
REQUEST_TIMEOUT = 30

# Seconds to wait between polling for job status.
POLL_SLEEP_TIME = 10

# Some forest loss events will not be successful in Studio due to not having enough
# Sentinel-2 images (e.g. recent events, or cloudy months for the monoculture model).
# So we lower the threshold to 50% of windows needing to succeed.
MIN_WINDOW_SUCCESS_RATE = 0.5


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
    # Number of workers to use for computing event areas when selecting events for
    # monoculture classification.
    monocrop_select_workers: int = 32


@dataclass
class RunPaths:
    """Paths relevant to different parts of a run."""

    # Initial alerts extracted from GLAD.
    initial_alerts_fname: UPath
    # Filename to store Studio job IDs (keyed by job name).
    job_ids_fname: UPath
    # Outputs from the forest loss driver Studio job.
    raw_studio_outputs_fname: UPath
    # Forest loss events after merging with previous events, but before adding
    # monoculture predictions.
    merged_events_fname: UPath
    # Request geometry (event centroids) for the monoculture Studio job.
    monocrop_request_fname: UPath
    # Outputs from the monoculture Studio job.
    monocrop_raw_outputs_fname: UPath
    # Filename to write all events (including monoculture predictions).
    all_events_fname: UPath
    # Directory to write per-country/month GeoJSON files.
    per_country_month_dir: UPath
    # Global (across runs) filename for latest events.
    # We use this to merge in events from the previous run.
    global_latest_fname: UPath


def _get_most_recent_friday() -> datetime:
    """Get the most recent Friday."""
    now = datetime.now(tz=UTC)
    friday = now - timedelta(days=(now.weekday() - 4) % 7)
    return friday


def simplify_features_to_centroids(
    features: list[dict],
) -> list[dict]:
    """Replace each feature's geometry with its centroid Point.

    Studio's FixedWindowPartitioner only uses the centroid to create a 128x128 window,
    so submitting Point geometries avoids issues with complex polygon geometries
    causing Studio job failures.

    Args:
        features: list of GeoJSON feature dicts (in WGS84).

    Returns:
        a new list of feature dicts with Point geometries at each original centroid.
    """
    simplified = []
    for feat in features:
        feat = copy.deepcopy(feat)
        shp = shapely.geometry.shape(feat["geometry"])
        centroid = shp.centroid
        feat["geometry"] = shapely.geometry.mapping(centroid)
        simplified.append(feat)
    return simplified


def start_studio_inference_job(
    client: StudioClient,
    job_name: str,
    model_id: str,
    features: list[dict],
    job_ids_fname: UPath,
) -> str:
    """Start an inference job on Studio, or return the already started job.

    The job IDs are cached in job_ids_fname keyed by the job name, so if a job with
    this name was previously started for this run, then the previous job ID is returned
    instead of starting a new job.

    Args:
        client: the Studio client to use.
        job_name: the name of the job, which must be unique within the run.
        model_id: the Studio model ID to run.
        features: the GeoJSON feature dicts (in WGS84) to run inference on.
        job_ids_fname: the filename caching the job IDs for this run.

    Returns:
        the Studio job ID.
    """
    # See if existing filename caching the job IDs exists.
    # If so, we load those already started jobs.
    if job_ids_fname.exists():
        with job_ids_fname.open() as f:
            job_ids_by_name = json.load(f)
    else:
        job_ids_by_name = {}

    if job_name in job_ids_by_name:
        logger.info(
            f"Found previously started job {job_ids_by_name[job_name]} named {job_name} in {job_ids_fname}"
        )
        return job_ids_by_name[job_name]  # type: ignore[no-any-return]

    # API request to run the model.
    logger.info(
        f"Starting a prediction job with {len(features)} features named {job_name}"
    )
    job_id = client.create_prediction(
        project_id=PROJECT_ID,
        model_id=model_id,
        name=job_name,
        geojson={
            "type": "FeatureCollection",
            "properties": {},
            "features": features,
        },
        min_window_success_rate=MIN_WINDOW_SUCCESS_RATE,
    )

    job_ids_by_name[job_name] = job_id
    with open_atomic(job_ids_fname, "w") as f:
        json.dump(job_ids_by_name, f)

    return job_id


def start_driver_inference_job(
    client: StudioClient, run_id: str, run_paths: RunPaths
) -> str:
    """Start the forest loss driver inference job on Studio.

    All of the forest loss events extracted from GLAD are submitted in one job.

    Args:
        client: the Studio client to use.
        run_id: the run ID.
        run_paths: the paths to use for this run.

    Returns:
        the Studio job ID.
    """
    # Read the GeoJSON, we will put it into the request.
    with run_paths.initial_alerts_fname.open() as f:
        geojson_data = json.load(f)

    # Simplify geometries to centroid points to avoid Studio failures due to
    # complex polygon geometries. The model only uses the centroid anyway.
    features = simplify_features_to_centroids(geojson_data["features"])

    return start_studio_inference_job(
        client=client,
        job_name=f"run_{run_id}",
        model_id=MODEL_ID,
        features=features,
        job_ids_fname=run_paths.job_ids_fname,
    )


def wait_for_studio_job(
    client: StudioClient, job_id: str, max_consecutive_errors: int = 3
) -> None:
    """Wait for a Studio prediction job to finish successfully.

    Raises an exception if the job fails.

    Args:
        client: the Studio client to use.
        job_id: the job ID to check.
        max_consecutive_errors: maximum consecutive connection/timeout/response-format
            errors before giving up.
    """
    consecutive_errors = 0
    while True:
        try:
            job_status = client.get_prediction(job_id)["status"]
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
        elif job_status != "completed":
            raise ValueError(
                f"expected status to be pending or completed, but got {job_status}"
            )

        break


def get_prediction_result(client: StudioClient, job_id: str) -> list[Feature]:
    """Get the FeatureCollection result from a Studio job.

    The prediction result is downloaded as a zip archive which, for our jobs, should
    contain a single GeoJSON file.

    Args:
        client: the Studio client to use.
        job_id: the Studio Prediction job ID.

    Returns:
        the list of features in the prediction result.
    """
    # Download the prediction result zip archive.
    url = client.get_prediction_result_url(job_id)
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_fname = UPath(tmp_dir) / "data.geojson"

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The download is a zip archive which for our jobs should contain a single
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


def get_cached_prediction_result(
    client: StudioClient, job_id: str, cache_fname: UPath
) -> list[Feature]:
    """Get the prediction result from a Studio job, caching it in cache_fname.

    Args:
        client: the Studio client to use.
        job_id: the Studio job ID to get results for.
        cache_fname: the filename to cache the downloaded result in.

    Returns:
        list of output features.
    """
    # See if this operation has been completed already.
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    if cache_fname.exists():
        logger.info(
            f"Loading previously downloaded Studio job outputs from {cache_fname}"
        )
        return vector_format.decode_from_file(cache_fname)

    logger.info(f"Getting outputs from job {job_id}")
    output_features = get_prediction_result(client, job_id)

    # Cache and return the features.
    vector_format.encode_to_file(cache_fname, output_features)
    return output_features


def add_input_properties_to_output_features(
    input_features: list[Feature],
    output_features: list[Feature],
) -> None:
    """Add properties and geometry from the corresponding input feature to each output.

    Matching is done by centroid proximity: for each output feature, we find the input
    feature whose WGS84 centroid is closest. This is robust to geometry changes (e.g.
    when output features have simplified Point geometries from Studio).

    The output feature's geometry is replaced with the matched input feature's geometry
    to restore the original polygon.

    Args:
        input_features: the original input features (superset of outputs).
        output_features: the output features from Studio to enrich.
    """
    centroid_index = CentroidIndex(input_features)

    # Match each output feature to the closest input feature by centroid distance.
    for output_feat in output_features:
        best_input_feat = centroid_index.find_closest(output_feat)
        if best_input_feat is None:
            raise ValueError(f"found no input feature for output feature {output_feat}")

        output_feat.properties.update(best_input_feat.properties)
        output_feat.geometry = best_input_feat.geometry


def merge_forest_loss_events(
    client: StudioClient,
    inference_job_id: str,
    asset_workers: int,
    run_paths: RunPaths,
) -> list[Feature]:
    """Get the forest loss events from Studio and merge them with previous events.

    Previous events are dropped if they start within the time window covered by the
    new events, or if more than OVERLAP_DEDUP_THRESHOLD of their area is covered by a
    new event (see filter_overlapping_previous_events).

    We also determine Sentinel-2 assets to display for the new events, and add an index
    property to all events.

    The whole operation is cached, so if it has already been completed we don't repeat
    it.

    Args:
        client: the Studio client to use.
        inference_job_id: the forest loss driver Studio job ID for this run.
        asset_workers: number of workers for getting Sentinel-2 assets.
        run_paths: paths to use for this run.

    Returns:
        list of merged forest loss events.
    """
    # See if this operation has been completed already.
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    if run_paths.merged_events_fname.exists():
        logger.info(
            f"Loading previously computed merged events from {run_paths.merged_events_fname}"
        )
        return vector_format.decode_from_file(run_paths.merged_events_fname)

    # Get prediction result from Studio.
    forest_loss_events = get_cached_prediction_result(
        client, inference_job_id, run_paths.raw_studio_outputs_fname
    )

    # Add back properties we had on our original features, like "country".
    # Also restore the original polygon geometries (Studio outputs have simplified
    # Point geometries since we submit centroids to avoid complex geometry issues).
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

        # First drop previous events within the time window covered by the new
        # events, since the new events supersede them.
        candidate_previous_events = [
            event
            for event in previous_events
            if event.properties["oe_start_time"] < earliest_start_time
        ]
        logger.info(
            f"Dropped {len(previous_events) - len(candidate_previous_events)} previous events with start time >= {earliest_start_time}"
        )

        # Then drop previous events that are mostly covered by a new event, since
        # those are most likely the same forest loss event detected again.
        kept_previous_events = filter_overlapping_previous_events(
            previous_events=candidate_previous_events,
            new_events=forest_loss_events,
        )
        forest_loss_events.extend(kept_previous_events)

        logger.info(f"After merging, this run has {len(forest_loss_events)} events")
    else:
        logger.info(
            f"Skipping merging since {run_paths.global_latest_fname} does not exist"
        )

    # Assign index property indicating index in the list.
    for index, event in enumerate(forest_loss_events):
        event.properties["index"] = index

    # Cache and return the events.
    vector_format.encode_to_file(run_paths.merged_events_fname, forest_loss_events)
    return forest_loss_events


def add_monoculture_predictions_from_studio(
    client: StudioClient,
    run_id: str,
    run_time: datetime,
    forest_loss_events: list[Feature],
    run_paths: RunPaths,
    select_workers: int,
) -> None:
    """Run the monoculture model on large-scale agriculture events and tag the events.

    The events that qualify (see select_monocrop_events) are submitted to Studio as
    centroids with time ranges covering the 12 monthly periods that the monoculture
    model expects. The predicted class is then added to the events as the
    monoculture_category property.

    The request geometry, job ID, and raw outputs are cached so the operation can be
    resumed if the pipeline restarts.

    Args:
        client: the Studio client to use.
        run_id: the run ID.
        run_time: the reference time of this run, used to compute event ages.
        forest_loss_events: the merged forest loss events, which are modified
            in-place.
        run_paths: paths to use for this run.
        select_workers: number of worker processes for computing event areas when
            selecting events.
    """
    logger.info("Selecting events for monoculture classification")
    selected_events = select_monocrop_events(
        forest_loss_events, run_time, workers=select_workers
    )
    logger.info(
        f"Selected {len(selected_events)} of {len(forest_loss_events)} events for monoculture classification"
    )
    if len(selected_events) == 0:
        logger.info("No events qualify for monoculture classification, skipping")
        return

    # Create (or load) the request geometry, cached for reproducibility.
    if run_paths.monocrop_request_fname.exists():
        logger.info(
            f"Using existing monoculture request geometry at {run_paths.monocrop_request_fname}"
        )
        with run_paths.monocrop_request_fname.open() as f:
            request_features = json.load(f)["features"]
    else:
        request_features = make_monocrop_request_features(selected_events, run_time)
        with open_atomic(run_paths.monocrop_request_fname, "w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "properties": {},
                    "features": request_features,
                },
                f,
            )

    job_id = start_studio_inference_job(
        client=client,
        job_name=f"run_{run_id}_monocrop",
        model_id=MONOCROP_MODEL_ID,
        features=request_features,
        job_ids_fname=run_paths.job_ids_fname,
    )
    logger.info(f"Got monoculture Studio job ID: {job_id}")
    wait_for_studio_job(client, job_id)

    output_features = get_cached_prediction_result(
        client, job_id, run_paths.monocrop_raw_outputs_fname
    )
    num_tagged = add_monoculture_predictions(selected_events, output_features)
    logger.info(
        f"Added monoculture predictions to {num_tagged} of {len(selected_events)} selected events"
    )


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
    2. Make OlmoEarth API request to run forest loss driver inference.
    3. Merge the new events with existing ones, dropping existing events that are
       within the new time window or that mostly overlap with a new event.
    4. Get metadata for Planetary Computer scenes that can be used to visualize the
       before and after images for each forest loss event.
    5. Make a second OlmoEarth API request to run the monoculture model on large-scale
       agriculture events from the last 12 months, and tag those events with the
       predicted monoculture category.
    6. Use tippecanoe to make tiles, and upload those tiles along with GeoJSON files to
       GCS so the website can access it.

    Args:
        integrated_config: the integrated configuration for all inference pipeline
            steps.
    """
    if MONOCROP_MODEL_ID is None:
        raise ValueError(
            "MONOCROP_MODEL_ID must be set to the Studio model ID of the monoculture model"
        )

    # Make run ID based on the current time.
    # We check the most recent Friday since that is when GLAD alerts are released, and
    # this way we will have the same run ID in case of restarts.
    # The run time (midnight UTC on that Friday) is also the reference time for
    # computing the age of forest loss events.
    run_time = _get_most_recent_friday().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    run_id = run_time.strftime("%Y%m%d")
    weka_ds_root = UPath(integrated_config.weka_base_dir) / f"dataset_{run_id}"
    gcs_ds_root = UPath(integrated_config.gcs_base_dir) / f"dataset_{run_id}"

    run_paths = RunPaths(
        initial_alerts_fname=weka_ds_root / "prediction_request_geometry.geojson",
        job_ids_fname=weka_ds_root / "job_id.json",
        raw_studio_outputs_fname=weka_ds_root / "events_from_studio_jobs.geojson",
        merged_events_fname=weka_ds_root / "merged_events.geojson",
        monocrop_request_fname=weka_ds_root / "monocrop_request_geometry.geojson",
        monocrop_raw_outputs_fname=weka_ds_root / "monocrop_events_from_studio.geojson",
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

    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    client = StudioClient.from_env()

    if run_paths.all_events_fname.exists():
        logger.info(
            f"Loading previously computed events from {run_paths.all_events_fname}"
        )
        forest_loss_events = vector_format.decode_from_file(run_paths.all_events_fname)
    else:
        # Start the forest loss driver Studio inference job and wait for it.
        inference_job_id = start_driver_inference_job(client, run_id, run_paths)
        logger.info(f"Got forest loss driver Studio job ID: {inference_job_id}")
        wait_for_studio_job(client, inference_job_id)

        # Get forest loss events from Studio, identify Sentinel-2 assets for
        # visualization for each event, and merge in previous events before the time
        # window we are processing.
        forest_loss_events = merge_forest_loss_events(
            client=client,
            inference_job_id=inference_job_id,
            asset_workers=integrated_config.asset_workers,
            run_paths=run_paths,
        )

        # Run the monoculture model on the large-scale agriculture events.
        add_monoculture_predictions_from_studio(
            client=client,
            run_id=run_id,
            run_time=run_time,
            forest_loss_events=forest_loss_events,
            run_paths=run_paths,
            select_workers=integrated_config.monocrop_select_workers,
        )

        # Save all_events.geojson in gcs_ds_root.
        vector_format.encode_to_file(run_paths.all_events_fname, forest_loss_events)

    # Write latest.geojson (used for merging) and per-country/month files.
    # We write latest.geojson here instead of when computing the events since it is a
    # cleaner way to ensure we always have completed this step (we don't want to write
    # all_events_fname and then job crashes and we don't write latest_events_fname).
    logger.info(
        f"Got {len(forest_loss_events)} after merging, writing latest.geojson and per-country/month files to GCS"
    )
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
