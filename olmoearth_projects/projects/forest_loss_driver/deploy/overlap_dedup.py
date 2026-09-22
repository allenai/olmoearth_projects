"""Spatial de-duplication of previous forest loss events against new events."""

import numpy as np
import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature

from olmoearth_projects.utils.logging import get_logger

logger = get_logger(__name__)

# A previous event is dropped if more than this fraction of its area is covered by a
# single new event.
OVERLAP_DEDUP_THRESHOLD = 0.5


def _get_wgs84_shapes(features: list[Feature]) -> np.ndarray:
    """Get the WGS84 shapely geometries of the features as an object array."""
    shapes = np.empty(len(features), dtype=object)
    for i, feat in enumerate(features):
        shapes[i] = feat.geometry.to_projection(WGS84_PROJECTION).shp
    return shapes


def filter_overlapping_previous_events(
    previous_events: list[Feature],
    new_events: list[Feature],
    overlap_threshold: float = OVERLAP_DEDUP_THRESHOLD,
) -> list[Feature]:
    """Drop previous events that are mostly covered by a new event.

    A previous event P is dropped if there is any new event N such that
    area(P intersect N) / area(P) > overlap_threshold. Each (P, N) pair is checked
    independently, i.e. the coverage from multiple new events is not summed.

    Areas are computed in WGS84 degrees. This is fine for the ratio since the
    distortion is locally uniform between a previous event and the new event that
    overlaps it.

    Candidate pairs are found with an STRtree over the new events, so only pairs with
    intersecting geometries are examined.

    Args:
        previous_events: the previous events to filter.
        new_events: the new events that may supersede previous events.
        overlap_threshold: the fraction of a previous event's area that must be
            covered by a single new event for the previous event to be dropped.

    Returns:
        the previous events that were not dropped, in their original order.
    """
    if len(previous_events) == 0 or len(new_events) == 0:
        return list(previous_events)

    prev_shapes = _get_wgs84_shapes(previous_events)
    new_shapes = _get_wgs84_shapes(new_events)

    # Bulk query yields a (2, num_pairs) array of (prev_idx, new_idx) pairs whose
    # geometries intersect.
    tree = shapely.STRtree(new_shapes)
    prev_idxs, new_idxs = tree.query(prev_shapes, predicate="intersects")

    drop_mask = np.zeros(len(previous_events), dtype=bool)
    if len(prev_idxs) > 0:
        prev_areas = shapely.area(prev_shapes[prev_idxs])
        intersection_areas = shapely.area(
            shapely.intersection(prev_shapes[prev_idxs], new_shapes[new_idxs])
        )
        # Previous events with zero area (degenerate geometries) are always kept.
        valid = prev_areas > 0
        ratios = np.zeros(len(prev_idxs), dtype=float)
        ratios[valid] = intersection_areas[valid] / prev_areas[valid]
        drop_mask[prev_idxs[ratios > overlap_threshold]] = True

    kept_events = [event for event, drop in zip(previous_events, drop_mask) if not drop]
    logger.info(
        f"Dropped {len(previous_events) - len(kept_events)} of {len(previous_events)} previous events that overlap more than {overlap_threshold:.0%} with a new event"
    )
    return kept_events
