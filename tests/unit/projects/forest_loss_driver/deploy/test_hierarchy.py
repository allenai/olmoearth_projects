import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry

from olmoearth_projects.projects.forest_loss_driver.deploy.hierarchy import (
    LEAVES,
    derive_categories,
    set_event_categories,
)


def make_probs(probs_by_leaf: dict[str, float]) -> list[float]:
    """Make a leaf probability list with the given probabilities (others zero)."""
    return [probs_by_leaf.get(leaf, 0.0) for leaf in LEAVES]


def test_single_leaf() -> None:
    props = derive_categories(make_probs({"selective_logging": 1.0}))
    assert props["driver_leaf"] == "selective_logging"
    assert props["driver_level1"] == "human_made"
    assert props["driver_level2"] == "forest_extraction"
    assert props["category"] == "logging"


def test_categories_use_summed_probabilities() -> None:
    # The top leaf is landslide, but the agriculture leaves together are more likely,
    # so the Level 1/2 nodes and legacy category come from agriculture.
    probs = make_probs(
        {
            "landslide": 0.4,
            "agriculture_small_scale": 0.3,
            "agriculture_pasture": 0.3,
        }
    )
    props = derive_categories(probs)
    assert props["driver_leaf"] == "landslide"
    assert props["driver_level1"] == "human_made"
    assert props["driver_level2"] == "agriculture"
    assert props["category"] == "agriculture"
    assert props["driver_probs"] == probs


def test_other_leaves_have_no_legacy_category() -> None:
    # natural_other has no legacy equivalent, so the legacy category is the most
    # likely of the remaining leaves.
    props = derive_categories(make_probs({"natural_other": 0.8, "flooding": 0.2}))
    assert props["driver_leaf"] == "natural_other"
    assert props["driver_level2"] == "natural_other"
    assert props["category"] == "river"


def test_set_event_categories() -> None:
    event = Feature(
        STGeometry(WGS84_PROJECTION, shapely.Point(0, 0), None),
        {
            "driver": "no_change",
            "probs": make_probs({"no_change": 0.9, "road": 0.1}),
            "country": "PE",
        },
    )
    set_event_categories(event)
    assert "driver" not in event.properties
    assert "probs" not in event.properties
    assert event.properties["driver_leaf"] == "no_change"
    assert event.properties["category"] == "none"
    assert event.properties["country"] == "PE"
