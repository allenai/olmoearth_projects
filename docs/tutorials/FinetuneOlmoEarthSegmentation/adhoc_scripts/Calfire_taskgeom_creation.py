#!/usr/bin/env python3
"""Compute bbox area & 10m pixel counts for task geometry creation."""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd
from pyproj import CRS
from shapely.errors import GEOSException
from shapely.geometry import box


def read_single_layer_gdb(gdb_path: str, layer: str | None) -> gpd.GeoDataFrame:
    """Read a single layer from a FileGDB, using the first layer if none specified."""
    if layer is None:
        layers = fiona.listlayers(gdb_path)
        if not layers:
            raise RuntimeError("No layers found in the FileGDB.")
        if len(layers) > 1:
            print(
                f"Warning: multiple layers found; using the first: {layers[0]}",
                file=sys.stderr,
            )
        layer = layers[0]
    return gpd.read_file(gdb_path, layer=layer)


def write_gdb(gdf: gpd.GeoDataFrame, out_gdb: str, layer_name: str) -> None:
    """Write a GeoDataFrame to a FileGDB, removing any existing GDB first."""
    out_path = Path(out_gdb)
    # Remove existing GDB (folder or file) if present
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    Path(out_gdb).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gdb, layer=layer_name)


def compute_bbox_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add bbox_area_m2, bbox_pix10m, max_side_pix10m, min_side_pix10m based on geometry envelopes."""

    def metrics_for_geom(geom: Any) -> tuple[float, int, int]:
        """Calculate area and pixel metrics for a geometry."""
        if geom is None or geom.is_empty:
            return 0.0, 0, 0  # area, max_side_pix10m, min_side_pix10m
        try:
            env = geom.envelope  # Polygon
            area = float(env.area)
            minx, miny, maxx, maxy = env.bounds
        except GEOSException:
            minx, miny, maxx, maxy = geom.bounds
            area = float(box(minx, miny, maxx, maxy).area)

        width_m = maxx - minx
        height_m = maxy - miny
        # Number of whole 10m pixels that fit along each side
        pix_w = int(math.floor(width_m / 10.0))
        pix_h = int(math.floor(height_m / 10.0))
        return area, max(pix_w, pix_h), min(pix_w, pix_h)

    out = gdf.copy()
    tupl = out.geometry.apply(metrics_for_geom)
    out["bbox_area_m2"] = tupl.apply(lambda t: t[0])
    out["bbox_pix10m"] = out["bbox_area_m2"] / 100.0
    out["max_side_pix10m"] = tupl.apply(lambda t: t[1])
    out["min_side_pix10m"] = tupl.apply(lambda t: t[2])
    return out


def apply_task_geometry_and_filter(
    gdf: gpd.GeoDataFrame, min_box_size_pix: int = 128
) -> gpd.GeoDataFrame:
    """Apply task geometry transformations and filter small polygons.

    Args:
        gdf: input GeoDataFrame with 'geometry' column in EPSG:32610 and bbox metrics computed.
        min_box_size_pix: minimum box size in 10m pixels (default: 128)

    Steps:
        0/ Create `task_geometry` = (geometry buffered 20 m).envelope
        1/ Drop polygons with bbox_pix10 < 1 (log count and %)
        2/ For rows with min_side_pix10m <= min_box_size_pix:
             task_geometry = geometry.buffer( (64 - min_side_pix10m + 2) * 10 )
    """
    out = gdf.copy()

    # --- 0) Filter out empty or invalid geometries before processing ---
    valid_mask = out.geometry.notna() & ~out.geometry.is_empty & out.geometry.is_valid
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"Warning: Dropping {n_invalid} rows with empty or invalid geometries")
        out = out.loc[valid_mask].copy()

    if len(out) == 0:
        raise ValueError("No valid geometries remaining after filtering")

    # --- 1) Initialize task_geometry with a 20 m buffer, then take the bbox (envelope) ---
    out["task_geom"] = out.geometry.buffer(20).envelope

    # --- 2) Drop polygons with bbox_pix10 < 1 (handle either bbox_pix10 or bbox_pix10m) ---
    bbox_col = "bbox_pix10" if "bbox_pix10" in out.columns else "bbox_pix10m"
    n_before = len(out)
    drop_mask = out[bbox_col] < 1
    n_drop = int(drop_mask.sum())
    pct_drop = (100.0 * n_drop / n_before) if n_before else 0.0

    print(
        f"Dropped {n_drop}/{gdf.shape[0]} ({pct_drop:.2f}%) polygons with 1 or less 10m pixel"
    )

    out = out.loc[~drop_mask].copy()

    # --- 3) For small boxes, grow the polygon so its shortest side fits min_box_size_pix pixels of 10 m
    small_mask = out["min_side_pix10m"] <= min_box_size_pix
    print(
        f"Growing task geometry for {small_mask.sum()} polygons with min side <= {min_box_size_pix} pixels"
    )
    if small_mask.any():
        missing = (min_box_size_pix - out.loc[small_mask, "min_side_pix10m"]).clip(
            lower=0
        )
        ceil_half_pix = missing.apply(lambda x: math.ceil(x / 2.0))
        grow_m = (ceil_half_pix + 2) * 10.0
        out.loc[small_mask, "task_geom"] = out.loc[small_mask, "geometry"].buffer(
            grow_m.clip(lower=0)
        )

    # Extra step to ensure geometries are valid
    out["task_geom"] = out["task_geom"].buffer(0)

    # Reproject → bbox → WKT (all on the temp series)
    tmp = gpd.GeoSeries(out["task_geom"], crs=out.crs)
    task_bbox_wgs84 = tmp.to_crs(4326).envelope

    # Validate that all task_geom are Polygon before converting to WKT
    # (envelope always produces Polygon, never MultiPolygon)
    invalid_geoms = task_bbox_wgs84[task_bbox_wgs84.geom_type != "Polygon"]
    if len(invalid_geoms) > 0:
        invalid_types = invalid_geoms.geom_type.value_counts()
        invalid_indices = invalid_geoms.index.tolist()
        # Get sample of actual geometries for debugging
        sample_geoms = invalid_geoms.head(3).to_wkt().tolist()
        raise ValueError(
            f"Found {len(invalid_geoms)} invalid task geometries that are not Polygon:\n"
            f"Geometry types: {dict(invalid_types)}\n"
            f"Row indices: {invalid_indices[:10]}{'...' if len(invalid_indices) > 10 else ''}\n"
            f"Sample geometries: {sample_geoms}\n"
            f"This should not happen after filtering. Please check input data."
        )

    # Converting to WKT to avoid multiple geometries in geodataframe
    out["task_geom"] = task_bbox_wgs84.to_wkt()

    return out


def main() -> None:
    """Main entry point for computing bbox area and pixel counts."""
    parser = argparse.ArgumentParser(
        description="Compute bbox area & 10m pixel counts; force CRS to EPSG:32610."
    )
    parser.add_argument("input_gdb", help="Path to the input FileGDB (e.g., data.gdb)")
    parser.add_argument(
        "-l",
        "--layer",
        default=None,
        help="Layer name in the GDB (optional; defaults to the first layer).",
    )
    parser.add_argument(
        "-o",
        "--output_gdb",
        default=None,
        help="Path to the output FileGDB (default: <input_basename>_bbox.gdb)",
    )
    parser.add_argument(
        "--min_box_size_pix",
        type=int,
        default=128,
        help="Minimum box size in 10m pixels (default: 128).",
    )
    args = parser.parse_args()

    in_gdb = args.input_gdb
    if not os.path.isdir(in_gdb) or not in_gdb.lower().endswith(".gdb"):
        parser.error("input_gdb must be a directory ending with .gdb")

    out_gdb = args.output_gdb or f"{in_gdb[:-4]}_bbox.gdb"

    # Read
    gdf = read_single_layer_gdb(in_gdb, args.layer)

    # --- Simple, forced projection: always reproject to EPSG:32610 ---
    # NOTE: .to_crs() requires a valid source CRS on gdf.crs.
    # If your GDB lacks a CRS, set it before reprojecting, e.g. gdf = gdf.set_crs(4326).to_crs(32610)
    gdf_32610 = gdf.to_crs(32610)

    # Compute metrics
    gdf_out = compute_bbox_metrics(gdf_32610)

    # Apply task-geometry + filtering logic
    gdf_out = apply_task_geometry_and_filter(gdf_out, args.min_box_size_pix)

    # Convert back to EPSG:4326
    gdf_out = gdf_out.to_crs(4326)

    # Choose output layer name
    out_layer = (
        args.layer
        if args.layer
        else (fiona.listlayers(in_gdb)[0] if fiona.listlayers(in_gdb) else "layer")
    )

    # Write
    write_gdb(gdf_out, out_gdb, out_layer)

    # Report
    crs_auth = None
    try:
        crs_auth = CRS.from_user_input(gdf_out.crs).to_authority()
    except Exception:
        # CRS authority extraction failed, will use fallback below
        crs_auth = None
    crs_str = (
        f"{crs_auth[0]}:{crs_auth[1]}"
        if crs_auth
        else (str(gdf_out.crs) if gdf_out.crs else "unknown")
    )
    print(f"✅ Wrote {out_layer!r} to: {out_gdb}")
    print(f"   CRS forced to: {crs_str}")
    print("   Added columns: 'bbox_area_m2' (float), 'bbox_pix10m' (float)")


if __name__ == "__main__":
    main()
