"""Forest loss driver label hierarchy, for postprocessing the driver model outputs.

The driver model (see olmoearth_run_data/forest_loss_driver/) predicts a probability
for each leaf of Amazon Conservation's three-level label hierarchy. This mirrors
rslp/forest_loss_driver/hierarchy.py in rslearn_projects, where the model is trained.

From the leaf probabilities we derive the Level 1 and Level 2 nodes, and the legacy
category (the flat categorization of the earlier driver models, which the web app's
driver legend and older events use), each as the node with the highest summed leaf
probability.
"""

from typing import Any

from rslearn.utils.feature import Feature

# Property written by the driver model with the top leaf, and its leaf probabilities.
MODEL_LEAF_PROPERTY = "driver"
MODEL_PROBS_PROPERTY = "probs"

# Properties set on each event from the driver model outputs.
LEAF_PROPERTY = "driver_leaf"
LEVEL1_PROPERTY = "driver_level1"
LEVEL2_PROPERTY = "driver_level2"
PROBS_PROPERTY = "driver_probs"
LEGACY_CATEGORY_PROPERTY = "category"

# Leaf -> (Level 1 node, Level 2 node), in the order of the model outputs.
LEAF_NODES: dict[str, tuple[str, str]] = {
    "agriculture_small_scale": ("human_made", "agriculture"),
    "agriculture_pasture": ("human_made", "agriculture"),
    "agriculture_soybean": ("human_made", "agriculture"),
    "agriculture_oil_palm": ("human_made", "agriculture"),
    "agriculture_rice": ("human_made", "agriculture"),
    "agriculture_other": ("human_made", "agriculture"),
    "mining_terrace": ("human_made", "mining"),
    "mining_riverine": ("human_made", "mining"),
    "airstrip": ("human_made", "infrastructures"),
    "road": ("human_made", "infrastructures"),
    "infrastructure_other": ("human_made", "infrastructures"),
    "logging_road": ("human_made", "forest_extraction"),
    "selective_logging": ("human_made", "forest_extraction"),
    "burned_area": ("human_made", "burned_area"),
    "human_other": ("human_made", "human_other"),
    "landslide": ("natural", "landslide"),
    "blowdown": ("natural", "blowdown"),
    "river_dynamics": ("natural", "river_dynamics"),
    "flooding": ("natural", "flooding"),
    "natural_other": ("natural", "natural_other"),
    "no_change": ("non_forest_loss", "no_change"),
    "already_altered": ("non_forest_loss", "already_altered"),
}
LEAVES: tuple[str, ...] = tuple(LEAF_NODES)

# Leaf -> legacy category. Leaves without a legacy equivalent (the "other" leaves)
# are omitted, so they don't count towards any legacy category.
LEAF_TO_LEGACY: dict[str, str] = {
    "agriculture_small_scale": "agriculture",
    "agriculture_pasture": "agriculture",
    "agriculture_soybean": "agriculture",
    "agriculture_oil_palm": "agriculture",
    "agriculture_rice": "agriculture",
    "agriculture_other": "agriculture",
    "mining_terrace": "mining",
    "mining_riverine": "mining",
    "airstrip": "airstrip",
    "road": "road",
    "logging_road": "road",
    "selective_logging": "logging",
    "burned_area": "burned",
    "landslide": "landslide",
    "blowdown": "hurricane",
    "river_dynamics": "river",
    "flooding": "river",
    "no_change": "none",
    "already_altered": "none",
}


def _argmax_summed(probs: list[float], node_of_leaf: dict[str, str]) -> str:
    """Get the node with the highest summed probability over its leaves."""
    sums: dict[str, float] = {}
    for leaf, prob in zip(LEAVES, probs):
        node = node_of_leaf.get(leaf)
        if node is None:
            continue
        sums[node] = sums.get(node, 0.0) + prob
    return max(sums, key=lambda node: sums[node])


def derive_categories(probs: list[float]) -> dict[str, Any]:
    """Get the leaf, Level 1/2 nodes and legacy category from the leaf probabilities.

    Args:
        probs: the leaf probabilities, in LEAVES order.

    Returns:
        dict of event properties to set.
    """
    if len(probs) != len(LEAVES):
        raise ValueError(f"expected {len(LEAVES)} leaf probabilities, got {len(probs)}")
    return {
        LEAF_PROPERTY: LEAVES[max(range(len(probs)), key=lambda i: probs[i])],
        LEVEL1_PROPERTY: _argmax_summed(
            probs, {leaf: nodes[0] for leaf, nodes in LEAF_NODES.items()}
        ),
        LEVEL2_PROPERTY: _argmax_summed(
            probs, {leaf: nodes[1] for leaf, nodes in LEAF_NODES.items()}
        ),
        LEGACY_CATEGORY_PROPERTY: _argmax_summed(probs, LEAF_TO_LEGACY),
        PROBS_PROPERTY: probs,
    }


def set_event_categories(event: Feature) -> None:
    """Replace the driver model output properties of an event with derived ones.

    The model's driver and probs properties are replaced by the driver_leaf,
    driver_level1, driver_level2, driver_probs and category properties.

    Args:
        event: the event with the driver model outputs, modified in-place.
    """
    probs = event.properties.pop(MODEL_PROBS_PROPERTY)
    event.properties.pop(MODEL_LEAF_PROPERTY, None)
    event.properties.update(derive_categories(probs))
