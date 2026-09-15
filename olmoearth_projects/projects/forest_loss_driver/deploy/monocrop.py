"""Monoculture classification of large-scale agriculture forest loss events.

After the forest loss driver model has classified the driver of each forest loss
event, we run a second model (see olmoearth_run_data/forest_loss_driver_monocrop/) on
the large-scale agriculture events to predict the type of agriculture (e.g. soybean or
oil palm).

This module contains the logic to select the events to process, to build the Studio
request geometry for them, and to add the predictions back onto the events.
"""

import copy
from datetime import datetime, timedelta

import shapely.geometry
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_crs

from olmoearth_projects.utils.logging import get_logger

from .centroid_index import CentroidIndex, get_wgs84_centroid

logger = get_logger(__name__)

# The forest loss driver category that qualifies events for monoculture
# classification.
AGRICULTURE_CATEGORY = "agriculture"

# Minimum area of events to run through the monoculture model. Smaller events are not
# "large-scale" agriculture and the model would be dominated by context outside the
# event anyway.
MIN_AREA_HA = 1.0

# Events must be at least this old (in days) so that there is at least one post-loss
# monthly mosaic. Otherwise there isn't enough imagery for an accurate prediction.
MIN_AGE_DAYS = 30

# Events must be younger than this (in days), i.e. from the most recent 12 months.
MAX_AGE_DAYS = 365

# The monoculture model inputs NUM_PERIODS monthly mosaics with PERIOD_DAYS days per
# period. The model is trained on (NUM_PERIODS - m) pre-loss periods followed by m
# post-loss periods for m from 1 to NUM_PERIODS, so we use the year ending at the run
# time for all events (see make_monocrop_request_features).
NUM_PERIODS = 12
PERIOD_DAYS = 30

# Property name on the Studio output features containing the predicted class.
STUDIO_CLASS_PROPERTY = "class_name"
STUDIO_PROBS_PROPERTY = "probs"
# Predictions of this class are treated as no prediction.
NODATA_CLASS = "nodata"

# Property names to set on the forest loss events.
MONOCULTURE_CATEGORY_PROPERTY = "monoculture_category"
MONOCULTURE_PROBS_PROPERTY = "monoculture_probs"


def get_area_ha(geometry: STGeometry) -> float:
    """Compute the area of the geometry in hectares.

    The geometry is projected to the UTM (or UPS) zone of its centroid at 1 m/pixel so
    that the area is in square meters.

    Args:
        geometry: the geometry.

    Returns:
        the area in hectares.
    """
    wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
    centroid = wgs84_geometry.shp.centroid
    utm_crs = get_utm_ups_crs(centroid.x, centroid.y)
    utm_geometry = wgs84_geometry.to_projection(Projection(utm_crs, 1, -1))
    return utm_geometry.shp.area / 10000


def select_monocrop_events(events: list[Feature], run_time: datetime) -> list[Feature]:
    """Select the forest loss events to run through the monoculture model.

    These are events that the forest loss driver model classified as agriculture, that
    are larger than MIN_AREA_HA, and that are between MIN_AGE_DAYS and MAX_AGE_DAYS old
    relative to run_time.

    Args:
        events: the merged forest loss events.
        run_time: the reference time of this run.

    Returns:
        the subset of events to process.
    """
    selected = []
    for event in events:
        if event.properties.get("category") != AGRICULTURE_CATEGORY:
            continue
        event_time = datetime.fromisoformat(event.properties["oe_start_time"])
        age_days = (run_time - event_time).days
        if age_days < MIN_AGE_DAYS or age_days >= MAX_AGE_DAYS:
            continue
        if get_area_ha(event.geometry) <= MIN_AREA_HA:
            continue
        selected.append(event)
    return selected


def make_monocrop_request_features(
    events: list[Feature], run_time: datetime
) -> list[dict]:
    """Create the Studio request features for the monoculture model.

    Each event becomes a Point feature at the event centroid (Studio only uses the
    centroid to create the fixed-size window).

    The model expects exactly NUM_PERIODS monthly periods, and is trained on a mix of
    pre-loss and post-loss months. Since we only run the model on events from the last
    year, we simply use the NUM_PERIODS periods ending at the run time as the
    oe_start_time/oe_end_time for all events: the event falls somewhere within the
    range, and the months after it are the post-loss months.

    Args:
        events: the events to process (from select_monocrop_events).
        run_time: the reference time of this run.

    Returns:
        list of GeoJSON feature dicts in WGS84.
    """
    start_time = run_time - timedelta(days=NUM_PERIODS * PERIOD_DAYS)
    end_time = run_time
    features = []
    for event in events:
        centroid = get_wgs84_centroid(event)
        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(centroid),
                "properties": {
                    "oe_start_time": start_time.isoformat(),
                    "oe_end_time": end_time.isoformat(),
                    # Retain the original event time and index for reference.
                    "event_time": event.properties["oe_start_time"],
                    "index": event.properties.get("index"),
                },
            }
        )
    return features


def add_monoculture_predictions(
    events: list[Feature], output_features: list[Feature]
) -> int:
    """Add the monoculture predictions from Studio onto the forest loss events.

    Each output feature is matched to the closest event by centroid, and that event's
    monoculture_category (and monoculture_probs) properties are set from the output.
    Events without a matching output (e.g. windows that failed due to insufficient
    imagery) are left unchanged, so if they had a prediction from a previous run then
    that prediction is retained.

    Args:
        events: the events that were submitted to the monoculture model, which are
            modified in-place.
        output_features: the output features from the monoculture Studio job.

    Returns:
        the number of events that were tagged with a prediction.
    """
    centroid_index = CentroidIndex(events)
    num_tagged = 0
    for output_feat in output_features:
        event = centroid_index.find_closest(output_feat)
        if event is None:
            logger.warning(f"found no event for monoculture output {output_feat}")
            continue

        class_name = output_feat.properties.get(STUDIO_CLASS_PROPERTY)
        if class_name is None or class_name == NODATA_CLASS:
            continue

        event.properties[MONOCULTURE_CATEGORY_PROPERTY] = class_name
        if STUDIO_PROBS_PROPERTY in output_feat.properties:
            event.properties[MONOCULTURE_PROBS_PROPERTY] = copy.deepcopy(
                output_feat.properties[STUDIO_PROBS_PROPERTY]
            )
        num_tagged += 1
    return num_tagged
