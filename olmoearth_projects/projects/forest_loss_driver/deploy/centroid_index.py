"""Spatial index for matching features by centroid proximity."""

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.grid_index import GridIndex

# Grid cell size in WGS84 degrees. 0.01 should give a reasonable number of grid cells
# (~100 pixels).
DEFAULT_GRID_SIZE = 0.01

# Default search radius in WGS84 degrees (~1 km).
DEFAULT_SEARCH_BUFFER = 0.01


def get_wgs84_centroid(feature: Feature) -> shapely.Point:
    """Get the centroid of the feature's geometry in WGS84."""
    return feature.geometry.to_projection(WGS84_PROJECTION).shp.centroid


class CentroidIndex:
    """Index features by their WGS84 centroid to find the closest feature to a point.

    We submit forest loss events to Studio as centroid Points, so the outputs from
    Studio need to be matched back to the events they came from. Matching by centroid
    proximity is robust to geometry changes (e.g. when output features have simplified
    Point geometries).
    """

    def __init__(
        self, features: list[Feature], grid_size: float = DEFAULT_GRID_SIZE
    ) -> None:
        """Create a new CentroidIndex.

        Args:
            features: the features to index. These are typically the forest loss
                events with their original polygon geometries; they are indexed by
                the centroid of the polygon (the same centroid that we submit to
                Studio as the request geometry).
            grid_size: the grid cell size in WGS84 degrees.
        """
        self.grid_index = GridIndex(grid_size)
        for feat in features:
            centroid = get_wgs84_centroid(feat)
            # Use centroid point as both the bounds key and the stored value.
            self.grid_index.insert(centroid.bounds, (centroid, feat))

    def find_closest(
        self, feature: Feature, search_buffer: float = DEFAULT_SEARCH_BUFFER
    ) -> Feature | None:
        """Find the indexed feature whose centroid is closest to the given feature's.

        Args:
            feature: the query feature. This is typically a Studio output feature,
                whose geometry is the Point (event centroid) that was submitted to
                Studio. Any geometry works though, since we compare centroids.
            search_buffer: only consider indexed features whose centroid is within
                this distance (in WGS84 degrees) along each axis.

        Returns:
            the closest indexed feature (i.e. the polygon event whose centroid matches
                the query point), or None if there is no feature within the search
                buffer.
        """
        query_centroid = get_wgs84_centroid(feature)
        search_bounds = (
            query_centroid.x - search_buffer,
            query_centroid.y - search_buffer,
            query_centroid.x + search_buffer,
            query_centroid.y + search_buffer,
        )
        candidates: list[tuple[shapely.Point, Feature]] = self.grid_index.query(
            search_bounds
        )

        best_feat: Feature | None = None
        best_distance: float | None = None
        for candidate_centroid, candidate_feat in candidates:
            distance = query_centroid.distance(candidate_centroid)
            if best_distance is None or distance < best_distance:
                best_feat = candidate_feat
                best_distance = distance

        return best_feat
