"""Point smoothing via Viterbi HMM.

Groups point detections across timesteps by spatial proximity, and then applies a
2-state HMM (present / absent) via the Viterbi algorithm. Applies NMS, and outputs
per-timestep smoothed GeoJSONs plus a history.geojson with start/end labels.

We re-project all points to their appropriate UTM zone at 10 m/pixel for spatial
matching and NMS. Viterbi is run as a single batched torch operation over all groups.
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from pyproj import Transformer
from rslearn.utils.get_utm_ups_crs import get_utm_ups_crs
from rslearn.utils.grid_index import GridIndex
from upath import UPath

from olmoearth_projects.utils.logging import get_logger

logger = get_logger(__name__)

# Resolution for pixel coordinates (meters per pixel).
METERS_PER_PIXEL = 10

# Matching distance threshold in pixels (200 m).
DISTANCE_THRESHOLD_PX = 200.0 / METERS_PER_PIXEL

# NMS distance in pixels (200 m).
NMS_DISTANCE_PX = 200.0 / METERS_PER_PIXEL

# Grid cell size for spatial indexing in pixels.
GRID_SIZE = 256

# Minimum number of valid timesteps for a group to be considered.
MIN_VALID_TIMESTEPS = 8

# Future label sentinel used when a group is still active at the last timestep.
FUTURE_LABEL = "2030-01"

# HMM parameters (2-state: absent=0, present=1).
INITIAL_PROBS = torch.tensor([0.5, 0.5], dtype=torch.float32)
TRANSITION_PROBS = torch.tensor(
    [
        # 5% chance to go from absent to present.
        [0.95, 0.05],
        # Marine infrastructure and wind turbines usually aren't torn down, so use a lower
        # 1% probability to go the other way.
        [0.01, 0.99],
    ],
    dtype=torch.float32,
)
EMISSION_PROBS = torch.tensor(
    [
        # 20% chance that the model made an error on any given timestep.
        [0.8, 0.2],
        [0.2, 0.8],
    ],
    dtype=torch.float32,
)


def _batch_lonlat_to_pixel(
    lons: list[float],
    lats: list[float],
) -> tuple[list[str], list[int], list[int]]:
    """Batch-convert lon/lat arrays to UTM pixel coordinates at 10 m/pixel.

    Groups points by their UTM/UPS zone and transforms each group together
    for efficiency.

    Returns:
        (proj_strs, cols, rows) — parallel lists aligned with input.
    """
    # Map from CRS string to the point indices that fall into that UTM/UPS zone.
    crs_groups: dict[str, list[int]] = defaultdict(list)
    for global_idx, (lon, lat) in enumerate(zip(lons, lats)):
        crs_str = get_utm_ups_crs(lon, lat).to_string()
        crs_groups[crs_str].append(global_idx)

    # Batch-transform each group of points sharing the same CRS.
    n = len(lons)
    proj_strs: list[str] = [""] * n
    cols: list[int] = [0] * n
    rows: list[int] = [0] * n
    for crs_str, indices in crs_groups.items():
        transformer = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        group_lons = np.array([lons[i] for i in indices])
        group_lats = np.array([lats[i] for i in indices])
        eastings, northings = transformer.transform(group_lons, group_lats)
        for group_idx, global_idx in enumerate(indices):
            proj_strs[global_idx] = crs_str
            cols[global_idx] = int(eastings[group_idx] / METERS_PER_PIXEL)
            rows[global_idx] = int(northings[group_idx] / -METERS_PER_PIXEL)

    return proj_strs, cols, rows


@dataclass
class Point:
    """A single detected point with its metadata."""

    lon: float
    lat: float

    # UTM pixel coordinates (computed from lon/lat).
    col: int
    row: int
    projection: str

    category: str
    score: float
    label: str


@dataclass
class Group:
    """A group of points across timesteps that likely represent the same object."""

    points: list[Point] = field(default_factory=list)

    def center_pixel(self) -> tuple[int, int]:
        """Average pixel coordinate across all points in the group."""
        col = sum(p.col for p in self.points) // len(self.points)
        row = sum(p.row for p in self.points) // len(self.points)
        return col, row

    @property
    def projection(self) -> str:
        """Projection string (from first point)."""
        return self.points[0].projection


def _parse_points(fc: dict, label: str) -> list[Point]:
    """Extract Point objects from a FeatureCollection dict.

    Re-projects each point's lon/lat into UTM pixel coordinates in batch.
    """
    features = fc.get("features", [])
    if not features:
        return []

    lons = [f["geometry"]["coordinates"][0] for f in features]
    lats = [f["geometry"]["coordinates"][1] for f in features]
    proj_strs, cols, rows = _batch_lonlat_to_pixel(lons, lats)

    points = []
    for i, feat in enumerate(features):
        points.append(
            Point(
                lon=lons[i],
                lat=lats[i],
                col=cols[i],
                row=rows[i],
                projection=proj_strs[i],
                category=feat["properties"]["category"],
                score=feat["properties"]["score"],
                label=label,
            )
        )
    return points


def _batched_viterbi(observations: torch.Tensor) -> torch.Tensor:
    """Run batched 2-state Viterbi over all groups simultaneously.

    Same structure as the raster Viterbi but over (T, N) instead of (T, H, W).

    Args:
        observations: (T, N) int tensor of observations (0=absent, 1=present).

    Returns:
        (T, N) int tensor of decoded states (0=absent, 1=present).
    """
    t_len, n_groups = observations.shape
    num_states = TRANSITION_PROBS.shape[0]

    # Forward pass.
    probs = INITIAL_PROBS.unsqueeze(0).expand(n_groups, -1).clone()  # (N, S)
    pointers = torch.zeros(t_len, n_groups, num_states, dtype=torch.long)

    for t in range(t_len):
        obs = observations[t]  # (N,)
        # Emission: select columns from EMISSION_PROBS for each observation value.
        # EMISSION_PROBS is (S, num_obs), index with obs -> (S, N), transpose -> (N, S).
        emit = EMISSION_PROBS.index_select(dim=1, index=obs).T  # (N, S)
        # Transition: (N, S_from, 1) * (S_from, S_to) -> (N, S_from, S_to).
        trans = probs[:, :, None] * TRANSITION_PROBS
        probs = trans.amax(dim=1) * emit  # (N, S)
        pointers[t] = trans.argmax(dim=1)  # (N, S)

    # Backward pass.
    cur = probs.argmax(dim=1)  # (N,)
    states_list: list[torch.Tensor] = [cur]
    for t in range(t_len - 1, 0, -1):
        cur = pointers[t][torch.arange(n_groups), cur]
        states_list.append(cur)
    states_list.reverse()

    return torch.stack(states_list, dim=0)  # (T, N)


def _extract_ranges(states: list[int]) -> list[tuple[int, int]]:
    """Extract contiguous runs of state==1 from a decoded state sequence.

    Returns:
        list of (start_idx, end_idx) tuples (end_idx exclusive).
    """
    ranges = []
    start_idx = -1
    for idx, state in enumerate(states):
        if state == 1 and start_idx == -1:
            start_idx = idx
        elif state == 0 and start_idx != -1:
            ranges.append((start_idx, idx))
            start_idx = -1
    if start_idx != -1:
        ranges.append((start_idx, len(states)))
    return ranges


def _apply_nms(groups: list[Group]) -> list[Group]:
    """Apply non-maximum suppression over groups in pixel coordinates.

    For each pair of groups within NMS_DISTANCE_PX (per-projection), the group
    with fewer points (or lower score as tiebreaker) is removed.

    Args:
        groups: list of groups to filter.

    Returns:
        Filtered list of groups.
    """
    nms_indexes: dict[str, GridIndex] = {}
    for group_idx, group in enumerate(groups):
        proj = group.projection
        if proj not in nms_indexes:
            nms_indexes[proj] = GridIndex(GRID_SIZE)
        last_point = group.points[-1]
        x, y = float(last_point.col), float(last_point.row)
        nms_indexes[proj].insert((x, y, x, y), group_idx)

    kept: list[Group] = []
    for group_idx, group in enumerate(groups):
        proj = group.projection
        last_point = group.points[-1]
        x, y = float(last_point.col), float(last_point.row)
        neighbors = nms_indexes[proj].query(
            (
                x - NMS_DISTANCE_PX,
                y - NMS_DISTANCE_PX,
                x + NMS_DISTANCE_PX,
                y + NMS_DISTANCE_PX,
            )
        )
        should_remove = False
        for other_idx in neighbors:
            if other_idx == group_idx:
                continue
            other = groups[other_idx]
            if other.projection != proj:
                continue
            other_last_point = other.points[-1]
            dx = last_point.col - other_last_point.col
            dy = last_point.row - other_last_point.row
            dist = math.sqrt(dx * dx + dy * dy)
            if dist >= NMS_DISTANCE_PX:
                continue
            # Remove if this group is worse (fewer points, or lower score).
            if len(group.points) < len(other.points):
                should_remove = True
                break
            if (
                len(group.points) == len(other.points)
                and last_point.score < other_last_point.score
            ):
                should_remove = True
                break
        if not should_remove:
            kept.append(group)

    return kept


def smooth_points(
    label: str,
    historical_dir: str,
    smoothed_dir: str,
) -> None:
    """Smooth point predictions across historical timesteps.

    Reads merged GeoJSON files from historical_dir (one per label like
    YYYY-MM.geojson), groups detections spatially across timesteps, applies
    Viterbi HMM, NMS, and writes smoothed per-timestep files + history.geojson.

    Args:
        label: the current label (most recent timestep).
        historical_dir: directory containing per-label merged GeoJSON files.
        smoothed_dir: directory to write smoothed outputs.
    """
    historical_upath = UPath(historical_dir)
    smoothed_upath = UPath(smoothed_dir)
    smoothed_upath.mkdir(parents=True, exist_ok=True)

    # Discover all available labels up to and including the requested label.
    labels: list[str] = []
    for fname in historical_upath.iterdir():
        if not fname.name.endswith(".geojson"):
            continue
        cur_label = fname.name.replace(".geojson", "")
        if cur_label > label:
            continue
        labels.append(cur_label)
    labels.sort()

    if label not in labels:
        raise ValueError(
            f"Label {label!r} not found in historical directory; available: {labels}"
        )

    logger.info("Found %d historical timesteps: %s", len(labels), labels)

    # Read all timesteps and build groups.
    # We process from most recent to oldest (most recent likely has best coverage).
    groups: list[Group] = []

    for ts_label in reversed(labels):
        fname = historical_upath / f"{ts_label}.geojson"
        with fname.open() as f:
            fc = json.load(f)

        cur_points = _parse_points(fc, ts_label)
        logger.info(
            "Matching %d groups with %d features at %s",
            len(groups),
            len(cur_points),
            ts_label,
        )

        # Build per-projection grid index for current points (in pixel coords).
        proj_indexes: dict[str, GridIndex] = {}
        for idx, point in enumerate(cur_points):
            if point.projection not in proj_indexes:
                proj_indexes[point.projection] = GridIndex(GRID_SIZE)
            x, y = float(point.col), float(point.row)
            proj_indexes[point.projection].insert((x, y, x, y), idx)

        # Match existing groups to current points.
        matched_indices: set[int] = set()
        for group in groups:
            proj = group.projection
            if proj not in proj_indexes:
                continue
            center_col, center_row = group.center_pixel()
            cx, cy = float(center_col), float(center_row)
            candidates = proj_indexes[proj].query(
                (cx - GRID_SIZE, cy - GRID_SIZE, cx + GRID_SIZE, cy + GRID_SIZE)
            )

            closest_idx = -1
            closest_dist = float("inf")
            for idx in candidates:
                if idx in matched_indices:
                    continue
                point = cur_points[idx]
                dx = center_col - point.col
                dy = center_row - point.row
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > DISTANCE_THRESHOLD_PX:
                    continue
                # Penalty for category mismatch rather than hard filter: e.g.
                # partially constructed wind turbines may be detected as platforms
                # and later as turbines once construction is done. We don't want
                # that to break Viterbi smoothing: marine infrastructure should
                # show up in the map even if we're unsure about the category. We will
                # use the latest predicted category as the category for the whole
                # group.
                if group.points[-1].category != point.category:
                    dist += DISTANCE_THRESHOLD_PX
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = idx

            if closest_idx >= 0:
                matched_indices.add(closest_idx)
                group.points.append(cur_points[closest_idx])

        # Unmatched points become new groups.
        for idx, point in enumerate(cur_points):
            if idx not in matched_indices:
                groups.append(Group(points=[point]))

    # Apply NMS over groups (in pixel coordinates, per-projection).
    logger.info("Applying NMS over %d groups", len(groups))
    groups = _apply_nms(groups)
    logger.info("NMS filtered to %d groups", len(groups))

    # Apply batched Viterbi over all groups at once.
    logger.info("Applying Viterbi to %d groups", len(groups))

    out_features: dict[str, list[dict]] = defaultdict(list)
    history_features: list[dict] = []

    if len(labels) < MIN_VALID_TIMESTEPS or not groups:
        logger.info("Too few timesteps or groups, skipping Viterbi")
    else:
        # Build (T, N) observation matrix: 1 if group detected at timestep, else 0.
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        obs = torch.zeros(len(labels), len(groups), dtype=torch.long)
        for g_idx, group in enumerate(groups):
            for point in group.points:
                t_idx = label_to_idx.get(point.label)
                if t_idx is not None:
                    obs[t_idx, g_idx] = 1

        # Run batched Viterbi.
        states = _batched_viterbi(obs)  # (T, N)

        # Extract ranges and build output features per group.
        for g_idx, group in enumerate(groups):
            group_states = states[:, g_idx].tolist()
            ranges = _extract_ranges(group_states)

            last_point = group.points[-1]
            for start_idx, end_idx in ranges:
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [last_point.lon, last_point.lat],
                    },
                    "properties": {
                        "category": last_point.category,
                        "score": last_point.score,
                    },
                }

                # Add to per-label outputs.
                for label_idx in range(start_idx, min(end_idx, len(labels))):
                    out_features[labels[label_idx]].append(feat)

                # Add to history with start/end.
                end_label = FUTURE_LABEL if end_idx >= len(labels) else labels[end_idx]
                score_avg = sum(p.score for p in group.points) / len(group.points)
                hist_feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [last_point.lon, last_point.lat],
                    },
                    "properties": {
                        "category": last_point.category,
                        "score": score_avg,
                        "start": labels[start_idx],
                        "end": end_label,
                    },
                }
                history_features.append(hist_feat)

    # Write per-label smoothed outputs.
    for ts_label, features in out_features.items():
        out_fname = smoothed_upath / f"{ts_label}.geojson"
        fc = {"type": "FeatureCollection", "features": features}
        with out_fname.open("w") as f:
            json.dump(fc, f)

    # Write history.
    history_fname = smoothed_upath / "history.geojson"
    history_fc = {"type": "FeatureCollection", "features": history_features}
    with history_fname.open("w") as f:
        json.dump(history_fc, f)

    logger.info(
        "Smoothing complete: %d timesteps, %d history features",
        len(out_features),
        len(history_features),
    )
