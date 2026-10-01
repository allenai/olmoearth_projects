from datetime import UTC, datetime, timedelta

import pytest
import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry

from olmoearth_projects.projects.forest_loss_driver.deploy.monocrop import (
    MONOCULTURE_CATEGORY_PROPERTY,
    MONOCULTURE_PROBS_PROPERTY,
    NUM_PERIODS,
    PERIOD_DAYS,
    add_monoculture_predictions,
    get_area_ha,
    make_monocrop_request_features,
    select_monocrop_events,
)

RUN_TIME = datetime(2026, 9, 11, tzinfo=UTC)

# Approximate size of one degree in meters near the equator.
METERS_PER_DEGREE = 111_000


def make_event(
    lon: float,
    lat: float,
    side_meters: float,
    age_days: int,
    category: str = "agriculture",
) -> Feature:
    """Make a square forest loss event with the given side length and age."""
    half_deg = side_meters / METERS_PER_DEGREE / 2
    shp = shapely.box(lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
    event_time = RUN_TIME - timedelta(days=age_days)
    return Feature(
        STGeometry(WGS84_PROJECTION, shp, None),
        {
            "category": category,
            "oe_start_time": event_time.isoformat(),
            "oe_end_time": event_time.isoformat(),
        },
    )


def test_get_area_ha() -> None:
    # 200 m x 200 m square is 4 ha.
    event = make_event(-74.5, -8.3, 200, 100)
    assert get_area_ha(event.geometry) == pytest.approx(4.0, rel=0.05)


def test_select_monocrop_events() -> None:
    large_agriculture = make_event(-74.5, -8.3, 200, 100)
    events = [
        large_agriculture,
        # Too small (0.25 ha).
        make_event(-74.6, -8.3, 50, 100),
        # Not agriculture.
        make_event(-74.7, -8.3, 200, 100, category="mining"),
        # Too recent.
        make_event(-74.8, -8.3, 200, 20),
        # Too old.
        make_event(-74.9, -8.3, 200, 400),
    ]
    assert select_monocrop_events(events, RUN_TIME) == [large_agriculture]


def test_select_monocrop_events_age_boundaries() -> None:
    youngest = make_event(-74.5, -8.3, 200, 30)
    oldest = make_event(-74.6, -8.3, 200, 364)
    events = [
        youngest,
        oldest,
        make_event(-74.7, -8.3, 200, 29),
        make_event(-74.8, -8.3, 200, 365),
    ]
    assert select_monocrop_events(events, RUN_TIME) == [youngest, oldest]


def test_select_monocrop_events_multiple_workers() -> None:
    # Alternate large and small events, with more events than one chunk, to check
    # that the order is preserved when computing areas across multiple workers.
    events = []
    for i in range(150):
        side_meters = 200 if i % 3 == 0 else 50
        events.append(make_event(-74.5 + i * 0.01, -8.3, side_meters, 100))
    expected = [event for i, event in enumerate(events) if i % 3 == 0]
    assert select_monocrop_events(events, RUN_TIME, workers=2) == expected


def test_make_monocrop_request_features() -> None:
    event = make_event(-74.5, -8.3, 200, 100)
    event.properties["index"] = 7
    (feat,) = make_monocrop_request_features([event], RUN_TIME)

    assert feat["geometry"]["type"] == "Point"
    assert feat["geometry"]["coordinates"] == pytest.approx((-74.5, -8.3))
    assert feat["properties"]["index"] == 7
    assert feat["properties"]["event_time"] == event.properties["oe_start_time"]

    start = datetime.fromisoformat(feat["properties"]["oe_start_time"])
    end = datetime.fromisoformat(feat["properties"]["oe_end_time"])
    assert end == RUN_TIME
    assert end - start == timedelta(days=NUM_PERIODS * PERIOD_DAYS)


def test_add_monoculture_predictions() -> None:
    event1 = make_event(-74.5, -8.3, 200, 100)
    event2 = make_event(-74.6, -8.3, 200, 100)
    event3 = make_event(-74.7, -8.3, 200, 100)
    event3.properties[MONOCULTURE_CATEGORY_PROPERTY] = "rice"

    def make_output(lon: float, lat: float, class_name: str) -> Feature:
        return Feature(
            STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None),
            {"class_name": class_name, "probs": [0.1, 0.9]},
        )

    outputs = [
        make_output(-74.5, -8.3, "soybean"),
        # nodata predictions are ignored.
        make_output(-74.6, -8.3, "nodata"),
        # event3 has no output this run (e.g. window failed), so it retains the
        # prediction from a previous run.
    ]
    num_tagged = add_monoculture_predictions([event1, event2, event3], outputs)

    assert num_tagged == 1
    assert event1.properties[MONOCULTURE_CATEGORY_PROPERTY] == "soybean"
    assert event1.properties[MONOCULTURE_PROBS_PROPERTY] == [0.1, 0.9]
    assert MONOCULTURE_CATEGORY_PROPERTY not in event2.properties
    assert event3.properties[MONOCULTURE_CATEGORY_PROPERTY] == "rice"
