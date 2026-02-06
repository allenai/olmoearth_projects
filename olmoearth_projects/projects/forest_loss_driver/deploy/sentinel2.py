"""Utilities for identifying Sentinel-2 assets to show for a forest loss event."""

import functools
import multiprocessing
from collections.abc import Callable
from datetime import datetime, timedelta
from multiprocessing.pool import IMapIterator
from typing import Any

import tqdm
from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.data_source import Item
from rslearn.data_sources.planetary_computer import Sentinel2
from rslearn.dataset.manage import retry
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry

# Duration and offset modifications to make to the forest loss event timestamp when
# looking for assets for visualization.
DURATION = timedelta(days=180)
PRE_OFFSET = timedelta(days=-300)
POST_OFFSET = timedelta(days=7)


@functools.cache
def _get_data_source() -> Sentinel2:
    """Get cached data source for identifying assets."""
    return Sentinel2(sort_by="eo:cloud_cover")


class StarImapWrapper:
    """Wrapper for a function to implement star_imap.

    A kwargs dict is passed to this wrapper, which then calls the underlying function
    with the unwrapped kwargs.
    """

    def __init__(self, fn: Callable[..., Any]):
        """Create a new StarImap.

        Args:
            fn: the underlying function to call.
        """
        self.fn = fn

    def __call__(self, kwargs: dict[str, Any]) -> Any:
        """Wrapped call to the underlying function.

        Args:
            kwargs: dict of keyword arguments to pass to the function.
        """
        return self.fn(**kwargs)


def star_imap(
    p: multiprocessing.pool.Pool,
    fn: Callable[..., Any],
    kwargs_list: list[dict[str, Any]],
) -> IMapIterator:
    """Wrapper for Pool.imap that exposes kwargs to the function.

    Args:
        p: the multiprocessing.pool.Pool to use.
        fn: the function to call, which accepts keyword arguments.
        kwargs_list: list of kwargs dicts to pass to the function.

    Returns:
        generator for outputs from the function in arbitrary order.
    """
    return p.imap(StarImapWrapper(fn), kwargs_list)


def _get_assets_for_feat(
    feat: Feature, num_assets: int
) -> tuple[list[dict], list[dict]]:
    """Get the Sentinel-2 assets for a feature.

    Args:
        feat: the feature to identify assets for.
        num_assets: the maximum number of pre and post assets to get.

    Returns: a tuple (pre_assets, post_assets) containing lists of Planetary Computer
        asset URLs for pre-event Sentinel-2 TCI images and post-event images.
    """
    data_source = _get_data_source()

    def get_assets_for_time_offset(
        offset: timedelta, duration: timedelta
    ) -> list[dict[str, Any]]:
        """Get assets after applying a time offset on the feature geometry."""
        ts = datetime.fromisoformat(feat.properties["oe_start_time"])
        geometry = STGeometry(
            feat.geometry.projection,
            feat.geometry.shp,
            (ts + offset, ts + offset + duration),
        )
        query_config = QueryConfig(
            space_mode=SpaceMode.CONTAINS,
            max_matches=num_assets,
        )
        # Run request with retries since Planetary Computer often has transient errors.
        item_groups: list[list[Item]] = retry(
            fn=lambda: data_source.get_items([geometry], query_config)[0],
            retry_max_attempts=10,
            retry_backoff=timedelta(seconds=5),
        )

        assets: list[dict] = []
        for item_group in item_groups:
            assert len(item_group) == 1
            item = item_group[0]
            assets.append(
                dict(
                    url=item.asset_urls["visual"],
                    ts=item.geometry.time_range[0].isoformat(),
                )
            )

        return assets

    pre_assets = get_assets_for_time_offset(PRE_OFFSET, DURATION)
    post_assets = get_assets_for_time_offset(POST_OFFSET, DURATION)
    return (pre_assets, post_assets)


def get_sentinel2_assets(
    features: list[Feature], workers: int, num_assets: int = 3
) -> None:
    """Identify suitable pre and post Sentinel-2 assets for each feature.

    The Planetary Computer asset URLs are added as properties to the feature. The
    assets are used for visualization in the website.

    Args:
        features: the forest loss event features. Their properties will be updated.
        workers: number of worker processes to use.
        num_assets: the maximum number of pre and post assets to get
    """
    p = multiprocessing.Pool(workers)
    get_assets_for_feat_args = [
        dict(
            feat=feat,
            num_assets=num_assets,
        )
        for feat in features
    ]
    outputs = tqdm.tqdm(
        star_imap(p, _get_assets_for_feat, get_assets_for_feat_args),
        desc="Identify pre/post Sentinel-2 assets",
        total=len(features),
    )
    for feat, (pre_assets, post_assets) in zip(features, outputs):
        feat.properties["pre_assets"] = pre_assets
        feat.properties["post_assets"] = post_assets
    p.close()
