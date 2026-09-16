import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry

from olmoearth_projects.projects.forest_loss_driver.deploy.overlap_dedup import (
    filter_overlapping_previous_events,
)


def make_box_event(
    min_x: float, min_y: float, max_x: float, max_y: float, name: str
) -> Feature:
    """Make a rectangular event with a name property for identification."""
    shp = shapely.box(min_x, min_y, max_x, max_y)
    return Feature(STGeometry(WGS84_PROJECTION, shp, None), {"name": name})


def make_point_event(x: float, y: float, name: str) -> Feature:
    """Make a degenerate (zero-area) point event."""
    return Feature(
        STGeometry(WGS84_PROJECTION, shapely.Point(x, y), None), {"name": name}
    )


def names(events: list[Feature]) -> list[str]:
    return [event.properties["name"] for event in events]


def test_fully_covered_previous_event_is_dropped() -> None:
    previous = [make_box_event(0, 0, 1, 1, "prev")]
    new = [make_box_event(-1, -1, 2, 2, "new")]
    assert names(filter_overlapping_previous_events(previous, new)) == []


def test_small_overlap_is_kept() -> None:
    # New event covers 25% of the previous event (half in x, half in y).
    previous = [make_box_event(0, 0, 1, 1, "prev")]
    new = [make_box_event(0.5, 0.5, 1.5, 1.5, "new")]
    assert names(filter_overlapping_previous_events(previous, new)) == ["prev"]


def test_threshold_is_strict() -> None:
    # New event covers exactly 50% of the previous event, which is not > 50%.
    previous = [make_box_event(0, 0, 1, 1, "prev")]
    new_half = [make_box_event(0.5, 0, 2, 1, "new")]
    assert names(filter_overlapping_previous_events(previous, new_half)) == ["prev"]

    # Slightly more than 50% is dropped.
    new_more = [make_box_event(0.4, 0, 2, 1, "new")]
    assert names(filter_overlapping_previous_events(previous, new_more)) == []


def test_non_intersecting_events_are_kept() -> None:
    previous = [
        # Touching along an edge only (zero-area intersection).
        make_box_event(0, 0, 1, 1, "adjacent"),
        # Far away.
        make_box_event(50, 50, 51, 51, "far"),
    ]
    new = [make_box_event(1, 0, 2, 1, "new")]
    assert names(filter_overlapping_previous_events(previous, new)) == [
        "adjacent",
        "far",
    ]


def test_empty_new_events_keeps_everything() -> None:
    previous = [make_box_event(0, 0, 1, 1, "a"), make_box_event(5, 5, 6, 6, "b")]
    assert names(filter_overlapping_previous_events(previous, [])) == ["a", "b"]
    assert filter_overlapping_previous_events([], previous) == []


def test_ratio_is_relative_to_previous_event() -> None:
    # A large new event covers a small previous event entirely, even though the
    # previous event is only a small fraction of the new event. The previous event is
    # dropped.
    small_previous = [make_box_event(0, 0, 0.1, 0.1, "small_prev")]
    large_new = [make_box_event(-1, -1, 1, 1, "large_new")]
    assert names(filter_overlapping_previous_events(small_previous, large_new)) == []

    # Conversely, a small new event inside a large previous event covers only a small
    # fraction of it, so the previous event is kept.
    large_previous = [make_box_event(-1, -1, 1, 1, "large_prev")]
    small_new = [make_box_event(0, 0, 0.1, 0.1, "small_new")]
    assert names(filter_overlapping_previous_events(large_previous, small_new)) == [
        "large_prev"
    ]


def test_coverage_from_multiple_new_events_is_not_summed() -> None:
    # Two new events each cover 40% of the previous event; neither alone exceeds 50%.
    previous = [make_box_event(0, 0, 1, 1, "prev")]
    new = [
        make_box_event(0, 0, 0.4, 1, "left"),
        make_box_event(0.6, 0, 1, 1, "right"),
    ]
    assert names(filter_overlapping_previous_events(previous, new)) == ["prev"]


def test_zero_area_previous_event_is_kept() -> None:
    previous = [make_point_event(0.5, 0.5, "point")]
    new = [make_box_event(0, 0, 1, 1, "new")]
    assert names(filter_overlapping_previous_events(previous, new)) == ["point"]


def test_order_is_preserved() -> None:
    previous = [
        make_box_event(0, 0, 1, 1, "a"),
        make_box_event(10, 10, 11, 11, "b"),
        make_box_event(20, 20, 21, 21, "c"),
        make_box_event(30, 30, 31, 31, "d"),
    ]
    # Drop b and c.
    new = [
        make_box_event(9, 9, 12, 12, "new1"),
        make_box_event(19, 19, 22, 22, "new2"),
    ]
    kept = filter_overlapping_previous_events(previous, new)
    assert names(kept) == ["a", "d"]
    # The returned objects are the original Feature instances.
    assert kept[0] is previous[0]
    assert kept[1] is previous[3]
