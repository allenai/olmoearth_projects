"""Test point smoothing via Viterbi HMM.

Creates four clusters across 12 quarterly timesteps:
1. Always present: detected every timestep -> should survive smoothing.
2. Appears halfway: detected in last 6 timesteps -> should survive from midpoint.
3. False positive: detected in only 1 timestep -> should be filtered out.
4. Hole: detected in all timesteps except 1 -> hole should be filled.
"""

import json
import tempfile

from upath import UPath

from olmoearth_projects.projects.satlas.smooth_points import smooth_points

# 12 quarterly labels spanning 3 years.
LABELS = [
    "2022-01",
    "2022-04",
    "2022-07",
    "2022-10",
    "2023-01",
    "2023-04",
    "2023-07",
    "2023-10",
    "2024-01",
    "2024-04",
    "2024-07",
    "2024-10",
]


def _make_feature(
    lon: float, lat: float, category: str = "platform", score: float = 0.9
) -> dict:
    """Create a GeoJSON point feature."""
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "category": category,
            "score": score,
        },
    }


def _find_cluster_feature(features: list[dict], lon: float, lat: float) -> dict | None:
    """Find the feature matching a lon/lat, or None if absent."""
    for feat in features:
        coords = feat["geometry"]["coordinates"]
        if abs(coords[0] - lon) < 0.01 and abs(coords[1] - lat) < 0.01:
            return feat
    return None


def test_smooth_points_four_clusters() -> None:
    """Test smoothing with four point clusters of varying temporal patterns."""
    # Cluster locations — spaced far enough apart (>> 200m) that they don't
    # interfere, and far enough in degrees that NMS doesn't merge them.
    always = (1.0, 50.0)
    halfway = (2.0, 51.0)
    false_pos = (3.0, 52.0)
    hole = (4.0, 53.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        historical_dir = UPath(tmp_dir) / "historical"
        smoothed_dir = UPath(tmp_dir) / "smoothed"
        historical_dir.mkdir()

        # Write per-timestep GeoJSON files.
        for label_idx, label in enumerate(LABELS):
            features = []

            # Cluster 1 (always): present in every timestep.
            features.append(_make_feature(*always))

            # Cluster 2 (halfway): present in last 6 timesteps (indices 6-11).
            if label_idx >= 6:
                features.append(_make_feature(*halfway))

            # Cluster 3 (false positive): present only at index 5.
            if label_idx == 5:
                features.append(_make_feature(*false_pos))

            # Cluster 4 (hole): present everywhere except index 6.
            if label_idx != 6:
                features.append(_make_feature(*hole))

            fc = {"type": "FeatureCollection", "features": features}
            with (historical_dir / f"{label}.geojson").open("w") as f:
                json.dump(fc, f)

        # Run smoothing.
        smooth_points(
            label="2024-10",
            historical_dir=str(historical_dir),
            smoothed_dir=str(smoothed_dir),
        )

        # Verify history.geojson exists.
        history_path = smoothed_dir / "history.geojson"
        assert history_path.exists(), "history.geojson should be created"
        with history_path.open() as f:
            history = json.load(f)

        history_features = history["features"]

        # 1. "always" cluster should appear in history with start at 2022-01.
        always_feat = _find_cluster_feature(history_features, *always)
        assert always_feat is not None, "Always-present cluster should be in history"
        assert always_feat["properties"]["start"] == "2022-01"

        # 2. "halfway" cluster should appear with start around 2023-07 (index 6).
        halfway_feat = _find_cluster_feature(history_features, *halfway)
        assert halfway_feat is not None, "Halfway cluster should survive smoothing"
        assert halfway_feat["properties"]["start"] >= "2023-04"
        assert halfway_feat["properties"]["start"] <= "2023-10"

        # 3. "false_pos" cluster: with only 1 detection out of 12 valid
        #    timesteps, Viterbi should filter it out (absent wins).
        false_pos_feat = _find_cluster_feature(history_features, *false_pos)
        assert false_pos_feat is None, (
            "False positive (single detection) should be smoothed away"
        )

        # 4. "hole" cluster should appear across all timesteps (hole filled).
        hole_feat = _find_cluster_feature(history_features, *hole)
        assert hole_feat is not None, "Hole cluster should survive smoothing"
        assert hole_feat["properties"]["start"] == "2022-01"
        # End should be the future sentinel (still active at last timestep).
        assert hole_feat["properties"]["end"] == "2030-01"

        # Also verify per-timestep smoothed outputs.
        # The "hole" cluster should be present even at the missing timestep.
        hole_missing_label = "2023-07"
        hole_ts_path = smoothed_dir / f"{hole_missing_label}.geojson"
        assert hole_ts_path.exists(), (
            f"Smoothed output for {hole_missing_label} should exist"
        )
        with hole_ts_path.open() as f:
            hole_ts_fc = json.load(f)
        assert _find_cluster_feature(hole_ts_fc["features"], *hole), (
            "Hole cluster should be present at the previously missing timestep"
        )

        # The "always" cluster should also be present at that timestep.
        assert _find_cluster_feature(hole_ts_fc["features"], *always), (
            "Always cluster should be present at every timestep"
        )
