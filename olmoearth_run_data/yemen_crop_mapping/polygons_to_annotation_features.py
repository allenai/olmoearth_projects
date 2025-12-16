"""
Convert a GeoJSON of labeled polygons (e.g. Dhamar.geojson) into the two files that
`olmoearth_run` expects for fine-tuning:

- annotation_task_features.geojson
- annotation_features.geojson

`olmoearth_run` expects:
- Each task is a GeoJSON Feature with:
  - properties.oe_annotations_task_id (UUID)
  - properties.oe_start_time / properties.oe_end_time (ISO-8601 datetimes)
  - geometry: Polygon/MultiPolygon (task boundary)
- Each annotation is a GeoJSON Feature with:
  - properties.oe_annotations_task_id (UUID) (must match some task above)
  - properties.oe_labels: dict[str, int|float|None] (we use {"category": class_id})
  - optional oe_start_time / oe_end_time
  - geometry: Polygon/MultiPolygon (annotation geometry; we clip to the task boundary)

This script grids the overall polygon extent into many "task windows" so that
`PolygonToRasterWindowPreparer` produces many raster-label windows (one per task).

Implementation note:
- We build the grid in WGS84 (lon/lat). The window size is specified in meters
  via (window_size_px * window_resolution_m) and converted to degrees using a
  simple meters-per-degree approximation at the bbox latitude. This is accurate
  enough for relatively small regions like a single governorate.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from shapely.geometry import mapping as shapely_mapping
from shapely.geometry import shape as shapely_shape
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry


# 9-class Yemen crop-type mapping.
# NOTE: These are 0-indexed class IDs (common for segmentation).
YEMEN_CROP_CLASSES: list[str] = [
    "orchards",
    "coffee",
    "inactive_cropland",
    "cereals",
    "not_cropland",
    "greenhouse",
    "fodder",
    "mixed_other",
    "qat",
]
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(YEMEN_CROP_CLASSES)}


def _parse_date_or_datetime(s: str) -> datetime:
    """
    Parse either "YYYY-MM-DD" or an ISO datetime.
    Always returns an aware datetime in UTC.
    """
    s = s.strip()
    # Common case in your GeoJSON: "2024-01-01"
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.replace(tzinfo=UTC)
    # Try ISO-8601 datetime
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _load_geojson_tolerant(path: Path) -> dict[str, Any]:
    """
    Load GeoJSON, tolerating a few bad prefixes (we've seen files starting with "am{...").
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.lstrip()
    if not raw.startswith("{"):
        idx = raw.find("{")
        if idx == -1:
            raise ValueError(f"{path} does not appear to contain JSON")
        raw = raw[idx:]
    return json.loads(raw)


@dataclass(frozen=True)
class LabeledPolygon:
    geom_wgs84: BaseGeometry
    class_id: int


def _iter_labeled_polygons(
    features: Iterable[dict[str, Any]],
    *,
    label_field: str,
) -> tuple[list[LabeledPolygon], datetime, datetime]:
    polys: list[LabeledPolygon] = []
    min_start: datetime | None = None
    max_end: datetime | None = None

    for feat in features:
        props = feat.get("properties") or {}
        label = props.get(label_field)
        if label not in CLASS_TO_ID:
            continue

        geom = shapely_shape(feat["geometry"])
        if geom.is_empty:
            continue
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        # Make valid if possible (self-intersections are common in hand-labeled polygons).
        try:
            from shapely import make_valid as _make_valid  # shapely>=2
        except Exception:  # pragma: no cover
            _make_valid = None
        if _make_valid is not None:
            geom = _make_valid(geom)
        if geom.is_empty:
            continue

        start_s = props.get("start_time")
        end_s = props.get("end_time")
        if isinstance(start_s, str):
            start_dt = _parse_date_or_datetime(start_s)
            min_start = start_dt if min_start is None else min(min_start, start_dt)
        if isinstance(end_s, str):
            end_dt = _parse_date_or_datetime(end_s)
            max_end = end_dt if max_end is None else max(max_end, end_dt)

        polys.append(LabeledPolygon(geom_wgs84=geom, class_id=CLASS_TO_ID[label]))

    if not polys:
        raise ValueError(f"No polygon features found with label_field={label_field} in {list(CLASS_TO_ID)}")

    # If times are missing, pick a safe default.
    if min_start is None:
        min_start = datetime(2024, 1, 1, tzinfo=UTC)
    if max_end is None:
        max_end = min_start

    return polys, min_start, max_end


def _bbox_union(polys: list[LabeledPolygon]) -> tuple[float, float, float, float]:
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    for p in polys:
        b = p.geom_wgs84.bounds
        minx = min(minx, b[0])
        miny = min(miny, b[1])
        maxx = max(maxx, b[2])
        maxy = max(maxy, b[3])
    return (minx, miny, maxx, maxy)

def _grid_cells_for_bbox(
    bbox_wgs84: tuple[float, float, float, float],
    *,
    window_size_px: int,
    window_resolution_m: float,
) -> list[tuple[int, int, int, int]]:
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    # Convert requested window size (meters) -> degrees at this latitude.
    center_lat = (min_lat + max_lat) / 2.0
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(center_lat))

    step_m = float(window_size_px) * float(window_resolution_m)
    step_lon = step_m / meters_per_deg_lon
    step_lat = step_m / meters_per_deg_lat

    # Snap to grid to keep stable boundaries.
    lon0 = math.floor(min_lon / step_lon) * step_lon
    lat0 = math.floor(min_lat / step_lat) * step_lat
    lon1 = math.ceil(max_lon / step_lon) * step_lon
    lat1 = math.ceil(max_lat / step_lat) * step_lat

    cells: list[tuple[int, int, int, int]] = []
    scale = 1_000_000  # store as ints to avoid float drift while iterating
    for lon in _frange(lon0, lon1, step_lon):
        for lat in _frange(lat0, lat1, step_lat):
            cells.append(
                (
                    int(round(lon * scale)),
                    int(round(lat * scale)),
                    int(round((lon + step_lon) * scale)),
                    int(round((lat + step_lat) * scale)),
                )
            )
    return cells


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    # Guard against floating point drift by counting steps.
    n = int(math.ceil((stop - start) / step))
    for i in range(n):
        yield start + i * step


def _build_strtree(polys: list[LabeledPolygon]) -> tuple[Any, list[BaseGeometry]]:
    """
    Build an STRtree if available (Shapely 2), else return (None, geoms).
    """
    geoms = [p.geom_wgs84 for p in polys]
    try:
        from shapely.strtree import STRtree  # type: ignore

        return STRtree(geoms), geoms
    except Exception:
        return None, geoms


def _query_candidates(tree: Any, geoms: list[BaseGeometry], query_geom: BaseGeometry) -> list[int]:
    if tree is None:
        # Brute force
        return list(range(len(geoms)))

    # Shapely 2: STRtree.query returns indices if built with geometries list.
    try:
        idxs = tree.query(query_geom)  # type: ignore[no-untyped-call]
        # idxs may already be a list[int] / ndarray[int]
        return list(map(int, idxs))
    except Exception:
        # Shapely 1 fallback: returns geometries
        hits = tree.query(query_geom)  # type: ignore[no-untyped-call]
        idx_map = {id(g): i for i, g in enumerate(geoms)}
        return [idx_map[id(g)] for g in hits if id(g) in idx_map]


def convert_polygons_to_annotation_features(
    *,
    input_geojson: Path,
    output_dir: Path,
    label_field: str,
    window_size_px: int,
    window_resolution_m: float,
    nodata_category_id: int,
    fill_empty_with_nodata: bool,
    override_start_time: datetime | None = None,
    override_end_time: datetime | None = None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer tolerant loader since some inputs have non-JSON prefixes.
    data = _load_geojson_tolerant(input_geojson)
    features = data.get("features") or []

    polys, inferred_start_dt, inferred_end_dt = _iter_labeled_polygons(features, label_field=label_field)
    start_dt = override_start_time or inferred_start_dt
    end_dt = override_end_time or inferred_end_dt
    if start_dt > end_dt:
        raise ValueError(
            f"start_time must be <= end_time. Got start_time={start_dt.isoformat()} end_time={end_dt.isoformat()}"
        )
    bbox = _bbox_union(polys)
    grid_cells = _grid_cells_for_bbox(
        bbox,
        window_size_px=window_size_px,
        window_resolution_m=window_resolution_m,
    )

    tree, tree_geoms = _build_strtree(polys)

    task_features: list[dict[str, Any]] = []
    annotation_features: list[dict[str, Any]] = []

    for cell_bounds in grid_cells:
        # Task geometry in WGS84 (GeoJSON expects lon/lat).
        scale = 1_000_000
        min_lon, min_lat, max_lon, max_lat = (b / scale for b in cell_bounds)
        cell_poly_wgs = shapely_box(min_lon, min_lat, max_lon, max_lat)

        candidate_idxs = _query_candidates(tree, tree_geoms, cell_poly_wgs)
        clipped_annos: list[dict[str, Any]] = []

        for i in candidate_idxs:
            poly = polys[i]
            if not poly.geom_wgs84.intersects(cell_poly_wgs):
                continue
            # Clip to the window so each task contains only shapes within its bounds.
            clipped = poly.geom_wgs84.intersection(cell_poly_wgs)
            if clipped.is_empty:
                continue

            clipped_annos.append(
                {
                    "type": "Feature",
                    "properties": {
                        # Filled after we generate the task_id below
                        "oe_labels": {"category": int(poly.class_id)},
                    },
                    "geometry": shapely_mapping(clipped),
                }
            )

        # If there are no polygon labels in this window, either skip it entirely
        # (default behavior), or optionally fill it with a nodata polygon so the
        # downstream window preparer still emits a raster window.
        if not clipped_annos and not fill_empty_with_nodata:
            continue

        task_id = uuid.uuid4()
        task_features.append(
            {
                "type": "Feature",
                "properties": {
                    "oe_annotations_task_id": str(task_id),
                    "oe_start_time": start_dt.isoformat().replace("+00:00", "Z"),
                    "oe_end_time": end_dt.isoformat().replace("+00:00", "Z"),
                    # Helpful provenance/debugging (olmoearth_run ignores extra properties).
                    "oe_window_size_px": int(window_size_px),
                    "oe_window_resolution_m": float(window_resolution_m),
                },
                "geometry": shapely_mapping(cell_poly_wgs),
            }
        )

        # If there are no polygon labels in this window, optionally add a single
        # "nodata" polygon covering the whole window so the downstream window preparer
        # still emits a full-sized raster label window (all nodata).
        if not clipped_annos and fill_empty_with_nodata:
            clipped_annos.append(
                {
                    "type": "Feature",
                    "properties": {
                        "oe_labels": {"category": int(nodata_category_id)},
                    },
                    "geometry": shapely_mapping(cell_poly_wgs),
                }
            )

        for anno in clipped_annos:
            anno["properties"]["oe_annotations_task_id"] = str(task_id)
            # We omit oe_start_time/oe_end_time for annotations; olmoearth_run uses task times.
            annotation_features.append(anno)

    task_path = output_dir / "annotation_task_features.geojson"
    anno_path = output_dir / "annotation_features.geojson"
    class_map_path = output_dir / "class_map.json"

    task_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": task_features}, indent=2),
        encoding="utf-8",
    )
    anno_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": annotation_features}, indent=2),
        encoding="utf-8",
    )
    class_map_path.write_text(json.dumps(CLASS_TO_ID, indent=2), encoding="utf-8")

    return task_path, anno_path, class_map_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input polygon GeoJSON (e.g. Dhamar.geojson)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory (will write annotation_features.geojson + annotation_task_features.geojson)",
    )
    parser.add_argument(
        "--label-field",
        default="crop_land",
        help="GeoJSON property containing the class name (default: crop_land)",
    )
    parser.add_argument(
        "--window-size-px",
        type=int,
        default=256,
        help="Task window size in pixels (default: 256)",
    )
    parser.add_argument(
        "--window-resolution-m",
        type=float,
        default=10.0,
        help="Meters per pixel (default: 10.0)",
    )
    parser.add_argument(
        "--nodata-category-id",
        type=int,
        default=10,
        help="Category ID to use for empty windows when --fill-empty-with-nodata is enabled (default: 10)",
    )
    parser.add_argument(
        "--fill-empty-with-nodata",
        action="store_true",
        help="If set, include empty windows by adding a nodata polygon covering the whole window.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        type=str,
        help=(
            "Optional override for all task oe_start_time values. "
            "Accepts YYYY-MM-DD or ISO datetime (e.g. 2024-01-01 or 2024-01-01T00:00:00Z)."
        ),
    )
    parser.add_argument(
        "--end-time",
        default=None,
        type=str,
        help=(
            "Optional override for all task oe_end_time values. "
            "Accepts YYYY-MM-DD or ISO datetime (e.g. 2024-01-31 or 2024-01-31T00:00:00Z)."
        ),
    )
    args = parser.parse_args()

    override_start = _parse_date_or_datetime(args.start_time) if args.start_time else None
    override_end = _parse_date_or_datetime(args.end_time) if args.end_time else None

    task_path, anno_path, class_map_path = convert_polygons_to_annotation_features(
        input_geojson=args.input,
        output_dir=args.output_dir,
        label_field=args.label_field,
        window_size_px=args.window_size_px,
        window_resolution_m=args.window_resolution_m,
        nodata_category_id=args.nodata_category_id,
        fill_empty_with_nodata=args.fill_empty_with_nodata,
        override_start_time=override_start,
        override_end_time=override_end,
    )

    print(f"Wrote {task_path}")
    print(f"Wrote {anno_path}")
    print(f"Wrote {class_map_path}")


if __name__ == "__main__":
    main()
