"""Tests for olmoearth_projects.projects.forest_loss_driver.deploy.sentinel2."""

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry

from olmoearth_projects.projects.forest_loss_driver.deploy.sentinel2 import (
    get_sentinel2_assets,
)


def test_get_sentinel2_assets() -> None:
    """Make sure get_sentinel2_assets returns assets."""
    # Create a feature corresponding to Seattle.
    geom = STGeometry(WGS84_PROJECTION, shapely.Point(-122.34, 47.64), None)
    feat = Feature(
        geom,
        dict(
            oe_start_time="2025-01-01T00:00:00Z",
            oe_end_time="2025-01-02T00:00:00Z",
        ),
    )
    # This call should fill in the pre_assets and post_assets properties.
    get_sentinel2_assets(features=[feat], workers=1, num_assets=3)
    # The feature was in the past and is covered by Sentinel-2 so it should get the max
    # number of assets (num_assets=3).
    pre_assets = feat.properties["pre_assets"]
    post_assets = feat.properties["post_assets"]
    assert len(pre_assets) == 3
    assert len(post_assets) == 3
