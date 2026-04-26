"""Tests for generate_tiles with 1x1 sub-features."""

from collections.abc import Callable
from unittest.mock import patch

from olmoearth_projects.projects.satlas.deploy import Application, generate_tiles


def _make_is_land(
    land_coords: set[tuple[float, float]],
) -> Callable[[float, float], bool]:
    """Return a mock is_land that returns True only for coords in land_coords."""

    def is_land(lat: float, lon: float) -> bool:
        return (lat, lon) in land_coords

    return is_land


class TestGenerateTilesSubFeatures:
    """Test that generate_tiles produces 1x1 sub-features per tile."""

    def test_land_app_has_sub_features(self) -> None:
        """Each tile for a land app should have a 'features' list."""
        tiles = generate_tiles(5, Application.WIND_TURBINE)
        for tile in tiles:
            assert "features" in tile
            assert len(tile["features"]) >= 1

    def test_sub_features_are_1x1(self) -> None:
        """Each sub-feature polygon should cover exactly 1x1 degree."""
        tiles = generate_tiles(5, Application.WIND_TURBINE)
        for tile in tiles:
            for feat in tile["features"]:
                coords = feat["geometry"]["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                width = max(lons) - min(lons)
                height = max(lats) - min(lats)
                assert width <= 1, f"sub-feature width {width} > 1"
                assert height <= 1, f"sub-feature height {height} > 1"

    def test_land_app_no_all_ocean_sub_features(self) -> None:
        """For land apps, no sub-feature should have all 4 corners on ocean."""
        from global_land_mask import globe

        tiles = generate_tiles(5, Application.WIND_TURBINE)
        for tile in tiles:
            for feat in tile["features"]:
                coords = feat["geometry"]["coordinates"][0]
                # First 4 coords are the corners (5th is closing repeat).
                corners = coords[:4]
                has_land = any(globe.is_land(lat, lon) for lon, lat in corners)
                assert has_land, f"land-app sub-feature has no land corners: {corners}"

    def test_marine_infra_no_all_land_sub_features(self) -> None:
        """For MARINE_INFRA, no sub-feature should have all 4 corners on land."""
        from global_land_mask import globe

        tiles = generate_tiles(5, Application.MARINE_INFRA)
        for tile in tiles:
            for feat in tile["features"]:
                coords = feat["geometry"]["coordinates"][0]
                corners = coords[:4]
                has_ocean = any(not globe.is_land(lat, lon) for lon, lat in corners)
                assert has_ocean, f"marine sub-feature has no ocean corners: {corners}"

    def test_tile_skipped_when_no_qualifying_cells(self) -> None:
        """A tile with no qualifying 1x1 cells should be skipped entirely."""
        # Mock: all ocean (no land anywhere).
        with patch(
            "olmoearth_projects.projects.satlas.deploy.globe.is_land",
            side_effect=_make_is_land(set()),
        ):
            tiles = generate_tiles(5, Application.WIND_TURBINE)
        assert len(tiles) == 0

    def test_tile_structure_preserved(self) -> None:
        """Tile-level properties and 5x5 geometry should be unchanged."""
        tiles = generate_tiles(5, Application.WIND_TURBINE)
        for tile in tiles:
            assert tile["type"] == "Feature"
            assert "tile_name" in tile["properties"]
            # The 5x5 geometry should span the tile size.
            coords = tile["geometry"]["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            width = max(lons) - min(lons)
            height = max(lats) - min(lats)
            assert width == 5 or width == 0  # 0 only at boundary 180
            assert height == 5 or height == 0  # 0 only at boundary 70
