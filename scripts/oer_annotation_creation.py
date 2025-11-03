#!/usr/bin/env python3
"""Create annotation task and annotation GeoJSONs from labeled geodata."""

import argparse
import json
import math
import sys
import uuid
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tqdm
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.geometry.base import BaseGeometry

_NUMERIC_NP_TYPES = (np.integer, np.floating)

# -------------------------------
# Column Name Utilities
# -------------------------------


def _normalize_column_name(name: str) -> str:
    """Normalize column name: lowercase, strip whitespace, replace common separators."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column_flexible(
    df: pd.DataFrame, preferred_names: list[str], context: str = "column"
) -> str | None:
    """Find a column in the DataFrame using case-insensitive and flexible matching.

    Args:
        df: DataFrame to search
        preferred_names: List of possible column names in order of preference
        context: Description for error messages

    Returns:
        The actual column name in the DataFrame, or None if not found
    """
    # Create a mapping of normalized names to actual column names
    col_map = {_normalize_column_name(col): col for col in df.columns}

    # Try each preferred name
    for name in preferred_names:
        normalized = _normalize_column_name(name)
        if normalized in col_map:
            actual_col = col_map[normalized]
            if actual_col != name:
                print(
                    f"Note: Using column '{actual_col}' for {context} (matched '{name}')",
                    file=sys.stderr,
                )
            return actual_col

    return None


def _ensure_column(
    df: pd.DataFrame,
    preferred_names: list[str],
    context: str = "column",
    required: bool = True,
) -> str | None:
    """Ensure a column exists, with helpful error messages if not.

    Args:
        df: DataFrame to search
        preferred_names: List of possible column names
        context: Description for error messages
        required: Whether to raise an error if not found

    Returns:
        The actual column name, or None if not found and not required
    """
    col = _find_column_flexible(df, preferred_names, context)

    if col is None and required:
        available = list(df.columns)
        raise KeyError(
            f"Could not find {context} column. Tried: {preferred_names}\n"
            f"Available columns: {available}\n"
            f"Column names are matched case-insensitively."
        )

    return col


# -------------------------------
# Helpers
# -------------------------------


def _coerce_to_utc_iso(
    dt: pd.Timestamp | datetime | str | int | float | None, ix: int
) -> str | None:
    """Parse any datetime-like value and return ISO 8601 string with timezone (UTC).

    If None, return None. Handles ISO strings, Unix timestamps, and pandas timestamps.
    """
    if dt is None or (isinstance(dt, float) and pd.isna(dt)) or dt == "NaT":
        print(f"Found null datetime value: {dt} for item {ix}", file=sys.stderr)
        return None

    # Handle Unix timestamps (seconds or milliseconds)
    if isinstance(dt, (int | float)) and not pd.isna(dt):
        # Heuristic: if value > year 3000 in seconds, it's probably milliseconds
        if dt > 32503680000:  # Year 3000 in seconds
            dt = dt / 1000.0
        try:
            ts = pd.to_datetime(dt, unit="s", utc=True)
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            # Fallback to default parsing
            ts = pd.to_datetime(dt, utc=True)
    else:
        ts = pd.to_datetime(dt, utc=True)  # assumes UTC if naive

    # Ensure timezone-aware UTC, then format as ISO string
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    # Use ISO 8601 with offset
    return ts.isoformat()


def _convert_label_to_int(
    gdf: gpd.GeoDataFrame, label_cols: list[str]
) -> gpd.GeoDataFrame:
    """Convert label value to integer if possible. Infer type from first occurence."""
    code_to_labels_dict = {}
    for col in label_cols:
        first_valid_idx = gdf[col].first_valid_index()
        if first_valid_idx is None:
            warnings.warn(f"Label column {col} has no valid values", UserWarning)
            continue

        first_valid = gdf[col].loc[first_valid_idx]
        if isinstance(first_valid, (str)):
            category = gdf[col].astype("category")
            codes = category.cat.codes  # ints; -1 marks NaN
            gdf[col] = codes
            if sum(gdf[col] == -1) > 0:
                warnings.warn(f"Label column {col} has null values", UserWarning)
            label_to_code = {cat: i for i, cat in enumerate(category.cat.categories)}
            code_to_label = {i: cat for i, cat in enumerate(category.cat.categories)}
            code_to_labels_dict[col] = code_to_label
            print(
                f"\nLabel encoding\n------------------\nLabel column: '{col}': {label_to_code}\n",
                file=sys.stderr,
            )

    return gdf, code_to_labels_dict


def _validate_time_range(start_iso: str | None, end_iso: str | None) -> None:
    """Matches validator semantics.

    - If either is None, allow it (for annotation feature).
    - If both present, enforce start <= end.
    """
    if start_iso is None or end_iso is None:
        return
    start = pd.to_datetime(start_iso)
    end = pd.to_datetime(end_iso)
    if not (start_iso) or not (end_iso):
        raise ValueError("Start time and end time cannot be empty")
    if start_iso.lower() == "nat" or end_iso.lower() == "nat":
        raise ValueError("Start time and end time cannot be NaT")
    if start > end:
        raise ValueError("Start time must be before end time")


def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame is in EPSG:4326. If no CRS, assume EPSG:4326."""
    if gdf.crs is None:
        warnings.warn("No CRS found in GeoDataFrame; assuming EPSG:4326.", UserWarning)
        gdf = gdf.set_crs(4326, allow_override=True)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def _coerce_numeric_or_none(val: Any) -> int | float | None:
    """Coerce value to numeric or None, treating pandas/numpy nulls as None."""
    # Treat pandas / numpy nulls as None
    if val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
        return None

    # Reject booleans (since bool is a subclass of int)
    if isinstance(val, bool):
        raise ValueError(
            "Boolean values are not valid labels; expected int/float/None."
        )

    # Python / NumPy numeric scalars
    if isinstance(val, (int, float) + _NUMERIC_NP_TYPES):
        return int(val) if isinstance(val, (int | np.integer)) else float(val)

    # Try to parse numeric strings
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return None
        # Prefer int when possible, else float
        try:
            i = int(s)
            return i
        except ValueError:
            try:
                f = float(s)
                return f
            except ValueError:
                raise ValueError(f"Non-numeric label string: {val!r}")

    # Any other type is invalid
    raise ValueError(f"Unsupported label type: {type(val).__name__}")


def _create_task_geom(
    id: str | int,
    polgeom: BaseGeometry,
    taskgeom: str | None,
    buffer: int = 100,
    taskgeom_crs: str | None = None,
) -> Polygon | MultiPolygon:
    """Ensure Polygon/MultiPolygon geometry for tasks.

    - If taskgeom_col, use that geometry as-is (transforming from taskgeom_crs to WGS84 if needed)
    - Otherwise: buffer the input geometry (polygon, linestring, or point) by the specified meters,
      and return the envelope (bounding box) of the buffered geometry.
    """
    # If taskgeom is provided, use it as-is
    if isinstance(taskgeom, str):
        custom_taskgeom = shapely.wkt.loads(taskgeom)

        # If taskgeom_crs is provided and different from WGS84, transform it
        if taskgeom_crs is not None:
            gdf_task = gpd.GeoDataFrame(geometry=[custom_taskgeom], crs=taskgeom_crs)
            if gdf_task.crs.to_epsg() != 4326:
                gdf_task = gdf_task.to_crs(4326)
                custom_taskgeom = gdf_task.geometry.iloc[0]
        else:
            # If no CRS provided, assume the taskgeom WKT is already in WGS84
            gdf_task = gpd.GeoDataFrame(geometry=[custom_taskgeom], crs=4326)
            custom_taskgeom = gdf_task.geometry.iloc[0]

        # Validate that the custom_taskgeom is a Polygon or MultiPolygon
        if not isinstance(custom_taskgeom, (Polygon | MultiPolygon)):
            raise ValueError(
                f"Task geometry from taskgeom_col must be a Polygon or MultiPolygon, "
                f"but got {custom_taskgeom.geom_type} for row id {id}. "
                f"custom geom found: {str(custom_taskgeom)}"
                f"Please ensure the task_geom column contains only Polygon or MultiPolygon geometries."
            )
        return custom_taskgeom

    # No taskgeom provided - create task geometry from input geometry
    if polgeom is None or polgeom.is_empty:
        raise ValueError("Encountered empty geometry; cannot build task geometry.")

    # For any geometry type (Polygon, LineString, Point, etc.):
    # 1. Project to 3857 for meter-based buffering
    # 2. Buffer by the specified distance in meters
    # 3. Get the envelope (bounding box) of the buffered geometry
    # 4. Reproject back to 4326

    gdf = gpd.GeoDataFrame(geometry=[polgeom], crs=4326).to_crs(3857)
    buffered = gdf.buffer(buffer, resolution=16)

    # Get the envelope (bounding box) of the buffered geometry
    envelope = buffered.envelope
    envelope_4326 = gpd.GeoDataFrame(geometry=envelope, crs=3857).to_crs(4326)

    envelope_geom = envelope_4326.geometry.iloc[0]

    # Envelope should always return a Polygon
    if isinstance(envelope_geom, Polygon):
        return envelope_geom

    raise ValueError(
        f"Expected Polygon from envelope, but got {envelope_geom.geom_type} for row id {id}."
    )


def _stable_task_uuid(source_id: str, namespace: uuid.UUID | None = None) -> uuid.UUID:
    """Stable UUIDv5 from an input identifier."""
    ns = namespace or uuid.NAMESPACE_URL
    name = f"oe-task:{source_id}"
    return uuid.uuid5(ns, name)


def _labels_dict(
    row: pd.Series, label_cols: list[str]
) -> dict[str, int | float | None]:
    """Build the oe_labels dict ensuring values are int|float|None and at least one is non-null."""
    labels: dict[str, int | float | None] = {}
    for col in label_cols:
        raw = row.get(col, None)
        labels[col] = _coerce_numeric_or_none(raw)

    if not any(v is not None for v in labels.values()):
        raise ValueError("At least one label must be non-null in oe_labels.")
    return labels


def _read_input(
    path: Path,
    layer: str | None,
    lat_col: str | None = None,
    lon_col: str | None = None,
    geom_col: str | None = None,
) -> gpd.GeoDataFrame:
    """Read GDB/SHP/GeoJSON using GeoPandas, or CSV using Pandas + points_from_xy.

    If a CSV is provided, latitude and longitude columns are auto-detected or can be specified.
    """
    suffix = path.suffix.lower()
    print(f"Reading file: {path}")

    if suffix == ".gdb":
        # For FileGDB, a layer is required if multiple layers exist.
        if layer is None:
            # Try reading all layers to guide the user
            layers = fiona.listlayers(path)
            if len(layers) != 1:
                raise ValueError(
                    f"GeoDatabase has {len(layers)} layers: {layers}. "
                    f"Please specify --layer."
                )
            layer = layers[0]
        gdf = gpd.read_file(path, layer=layer)

    elif suffix == ".csv":
        df = pd.read_csv(path)

        # Auto-detect coordinate columns with flexible matching
        lat_candidates = (
            [lat_col]
            if lat_col
            else ["latitude", "lat", "y", "northing", "Latitude", "LAT", "Y"]
        )
        lon_candidates = (
            [lon_col]
            if lon_col
            else [
                "longitude",
                "lon",
                "lng",
                "long",
                "x",
                "easting",
                "Longitude",
                "LON",
                "LNG",
                "X",
            ]
        )

        lat_name = _ensure_column(df, lat_candidates, "latitude", required=True)
        lon_name = _ensure_column(df, lon_candidates, "longitude", required=True)

        print(f"Using coordinates: {lon_name}, {lat_name}", file=sys.stderr)
        geom = gpd.points_from_xy(df[lon_name], df[lat_name], crs=4326)
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
    else:
        gdf = gpd.read_file(path)

        # Handle custom geometry column name
        if geom_col and geom_col in gdf.columns and geom_col != "geometry":
            gdf = gdf.set_geometry(geom_col)
            print(f"Using geometry column: {geom_col}", file=sys.stderr)

    if gdf.crs != "EPSG:4326":
        print(f"Converting CRS from {gdf.crs} to EPSG:4326", file=sys.stderr)
        gdf = gdf.to_crs(4326)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
        print("Input had no CRS; assuming EPSG:4326.", file=sys.stderr)
    return gdf


# -------------------------------
# Core logic
# -------------------------------


def build_feature_collections(
    gdf: gpd.GeoDataFrame,
    id_col: str = "id",
    start_col: str = "start_time",
    end_col: str = "end_time",
    taskgeom_col: str | None = None,
    label_cols: list[str] | None = None,
    taskgeom_crs: str | None = None,
    buffer: int = 100,
) -> tuple[dict, dict]:
    """Construct olmoearth_run compatible FeatureCollections for tasks and annotations.

    Args:
        gdf: Input GeoDataFrame
        id_col: Column name for unique identifiers
        start_col: Column name for start time
        end_col: Column name for end time
        taskgeom_col: Column name for custom task geometry (as WKT strings)
        label_cols: List of columns to use as labels
        taskgeom_crs: CRS of the task_geom WKT geometries (if different from main geometry)
        buffer: Buffer distance in meters for creating task geometries

    Returns:
        (task_fc, annotation_fc) as GeoJSON-ready dicts.
    """
    # Store original CRS before transformation if taskgeom_crs not explicitly provided
    if taskgeom_crs is None and taskgeom_col is not None and gdf.crs is not None:
        print(
            f"Assuming task geometry CRS is the same as the input geometry CRS: {gdf.crs.to_string()}",
            file=sys.stderr,
        )
        taskgeom_crs = gdf.crs.to_string()

    gdf = _ensure_wgs84(gdf)

    # Flexible column matching
    id_candidates = [
        id_col,
        "id",
        "ID",
        "unique_id",
        "UniqueID",
        "objectid",
        "OBJECTID",
        "fid",
        "FID",
    ]
    start_candidates = [
        start_col,
        "start_time",
        "start_date",
        "starttime",
        "startdate",
        "StartTime",
        "START_TIME",
    ]
    end_candidates = [
        end_col,
        "end_time",
        "end_date",
        "endtime",
        "enddate",
        "EndTime",
        "END_TIME",
    ]

    # Find actual column names
    actual_id_col = _ensure_column(gdf, id_candidates, "ID", required=False)
    actual_start_col = _ensure_column(
        gdf, start_candidates, "start time", required=True
    )
    actual_end_col = _ensure_column(gdf, end_candidates, "end time", required=True)

    # Auto-generate IDs if not present
    if actual_id_col is None:
        print("Warning: No ID column found. Auto-generating IDs.", file=sys.stderr)
        gdf["_auto_id"] = [f"auto_{i:06d}" for i in range(len(gdf))]
        actual_id_col = "_auto_id"

    # Handle labels flexibly
    if not label_cols:
        # Try to find a label column
        label_candidates = ["label", "Label", "class", "Class", "category", "Category"]
        actual_label_col = _find_column_flexible(gdf, label_candidates, "label")
        if actual_label_col:
            label_cols = [actual_label_col]
        else:
            raise ValueError(
                "No label columns specified and none found. "
                "Please specify --label-cols or ensure your data has a 'label' column.\n"
                f"Available columns: {list(gdf.columns)}"
            )
    else:
        # Validate user-specified label columns exist
        actual_label_cols: list[str] = []
        for lc in label_cols:
            col = _ensure_column(gdf, [lc], f"label column '{lc}'", required=True)
            if col is not None:
                actual_label_cols.append(col)
        label_cols = actual_label_cols

    # Handle taskgeom column
    actual_taskgeom_col = None
    if taskgeom_col:
        actual_taskgeom_col = _ensure_column(
            gdf, [taskgeom_col], "task geometry", required=True
        )

    # Convert string labels to integer as per olmoearth_run requirement
    gdf, code_to_labels_dict = _convert_label_to_int(gdf, label_cols)

    task_features = []
    annotation_features = []

    for idx, row in tqdm.tqdm(gdf.iterrows(), total=len(gdf), desc="Processing rows"):
        anot_geom: BaseGeometry = row.geometry
        taskgeom: str | None = row[actual_taskgeom_col] if actual_taskgeom_col else None

        if anot_geom is None or anot_geom.is_empty:
            raise ValueError(f"Row {idx}: empty geometry.")

        source_id = str(row[actual_id_col])
        task_uuid = _stable_task_uuid(source_id)

        # Times
        start_iso = _coerce_to_utc_iso(row[actual_start_col], idx)
        end_iso = _coerce_to_utc_iso(row[actual_end_col], idx)

        # Validate time range per validator rules
        _validate_time_range(start_iso, end_iso)

        # Labels dict (must have at least one non-null)
        labels = _labels_dict(row, label_cols)
        src_labels = {k: code_to_labels_dict[k].get(v, v) for k, v in labels.items()}

        # ---------------- Task Feature ----------------
        task_geom = _create_task_geom(
            row[actual_id_col],
            anot_geom,
            taskgeom,
            buffer=buffer,
            taskgeom_crs=taskgeom_crs,
        )
        task_feature = {
            "type": "Feature",
            "id": None,
            "bbox": None,
            "geometry": mapping(task_geom),
            "properties": {
                "src_feature_id": str(source_id),
                "oe_annotations_task_id": str(task_uuid),
                "oe_start_time": start_iso,
                "oe_end_time": end_iso,
            },
        }
        task_features.append(task_feature)

        # ---------------- Annotation Feature ----------------
        # Geometry can be any valid geometry; use the original geometry.
        ann_feature = {
            "type": "Feature",
            "id": None,
            "bbox": None,
            "geometry": mapping(anot_geom),
            "properties": {
                "src_feature_id": str(source_id),
                "oe_annotations_task_id": str(task_uuid),
                "oe_start_time": start_iso,  # same as task (per requirement)
                "oe_end_time": end_iso,  # same as task (per requirement)
                "oe_labels": labels,
                "src_labels": src_labels,
            },
        }
        annotation_features.append(ann_feature)

    task_fc = {
        "type": "FeatureCollection",
        "bbox": None,
        "features": task_features,
    }
    ann_fc = {
        "type": "FeatureCollection",
        "bbox": None,
        "features": annotation_features,
    }
    return task_fc, ann_fc


def write_geojson(obj: dict, out_path: Path) -> None:
    """Write a GeoJSON object to a file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------------
# CLI
# -------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Build annotation task & annotation GeoJSONs from a GIS file or CSV.",
        epilog="Column names are matched case-insensitively with common variations.",
    )
    p.add_argument(
        "input", type=Path, help="Path to .gdb, .shp, .geojson, or .csv file."
    )
    p.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer name (required if input is a .gdb with multiple layers).",
    )
    p.add_argument(
        "--lat-col",
        type=str,
        default=None,
        help="Latitude column name for CSV input. Auto-detected if not specified.",
    )
    p.add_argument(
        "--lon-col",
        type=str,
        default=None,
        help="Longitude column name for CSV input. Auto-detected if not specified.",
    )
    p.add_argument(
        "--geom-col",
        type=str,
        default=None,
        help="Geometry column name if different from 'geometry'.",
    )
    p.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="Column containing unique identifier per feature. Auto-generated if not found. Default: id",
    )
    p.add_argument(
        "--start-col",
        type=str,
        default="start_time",
        help="Column for start time. Default: start_time",
    )
    p.add_argument(
        "--end-col",
        type=str,
        default="end_time",
        help="Column for end time. Default: end_time",
    )
    p.add_argument(
        "--label-cols",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more columns to include in oe_labels dict. "
            "Auto-detected if not specified (looks for 'label', 'class', 'category')."
        ),
    )
    p.add_argument(
        "--taskgeom-col",
        type=str,
        default=None,
        help=("Column containing task geometry (as WKT strings). "),
    )
    p.add_argument(
        "--taskgeom-crs",
        type=str,
        default=None,
        help=(
            "CRS/EPSG code for task_geom WKT geometries if different from main geometry. "
            "Examples: 'EPSG:26910', 'EPSG:32610'. If not provided, assumes same CRS as input data."
        ),
    )
    p.add_argument(
        "--buffer",
        type=int,
        default=100,
        help=(
            "Buffer distance in meters for creating task geometries from input features. "
            "The input geometry is buffered by this distance, and the envelope (bounding box) "
            "of the buffered geometry is used as the task geometry. Default: 100 meters."
        ),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Output directory. Default: current directory.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and column mappings without writing output files.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Main entry point for creating annotation task and annotation GeoJSONs."""
    args = parse_args(argv)

    try:
        # Read
        gdf = _read_input(
            args.input,
            args.layer,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            geom_col=args.geom_col,
        )

        print(f"\nSuccessfully loaded {len(gdf)} features")
        print(f"Available columns: {list(gdf.columns)}")

        if args.dry_run:
            print("\n--dry-run mode: Validating column mappings...")

            # Test column detection

            # Capture stderr to see what columns are being used
            print("\nColumn Detection Results:")
            print("-" * 50)

        # Build FeatureCollections
        task_fc, ann_fc = build_feature_collections(
            gdf=gdf,
            id_col=args.id_col,
            start_col=args.start_col,
            end_col=args.end_col,
            label_cols=args.label_cols,
            taskgeom_col=args.taskgeom_col,
            taskgeom_crs=args.taskgeom_crs,
            buffer=args.buffer,
        )

        if args.dry_run:
            print(
                "\n✓ Validation successful! All required columns found and data is valid."
            )
            print(
                f"  Would create {len(task_fc['features'])} tasks and {len(ann_fc['features'])} annotations."
            )
            return 0

        # Write
        out_tasks = args.outdir / "annotation_task_features.geojson"
        out_annotations = args.outdir / "annotation_features.geojson"

        write_geojson(task_fc, out_tasks)
        write_geojson(ann_fc, out_annotations)

        print("\n✓ Success!")
        print(f"  Wrote: {out_tasks}")
        print(f"  Wrote: {out_annotations}")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if "--dry-run" in sys.argv or args.dry_run:
            print("\nTip: Check your column names and data format.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
