"""Generate landslide and no-landslide label polygons for OlmoEarth fine-tuning.

This script processes landslide inventory data to create training labels for
landslide detection models. It generates:

1. Landslide polygons (positive samples) from inventory data
2. No-landslide polygons (negative samples) using the same geometry 1 month earlier

The script filters to high-confidence events, creates negative samples by
temporal offset, removes overlaps where other landslides occurred, applies
geographic balancing, and computes task geometries.

Example:
    $ python prepare_labels.py --input-zip olmoearth_run_data/landslides/inventories.zip
"""

import argparse
import logging
import math
import multiprocessing
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from rslearn.utils import get_utm_ups_crs
from shapely.errors import GEOSException
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

GEO_CRS = "EPSG:4326"
DEFAULT_DATA_DIR = Path("olmoearth_run_data/landslides")
DEFAULT_OUTPUT_NAME = "landslide_labels.json"

# Data filtering constants
MIN_EVENT_CONFIDENCE = 1.0
SHAPEFILE_NAME = "inventories.shp"

# Label constants
LABEL_LANDSLIDE = "with_landslide"
LABEL_NO_LANDSLIDE = "without_landslide"

# Column name constants
COL_LABEL = "label"
COL_EVENT_CONF = "event_conf"
COL_EVENT_DATE = "event_date"
COL_LOCATION = "location"
COL_START_TIME = "start_time"
COL_END_TIME = "end_time"
COL_GEOMETRY = "geometry"
COL_TASK_GEOM = "task_geom"
COL_POLYGON_ID = "polygon_id"
COL_GEOM_CRS = "COL_GEOM_CRS"

# Time constants (in days)
NEGATIVE_SAMPLE_OFFSET_DAYS = 30  # Negative samples are 1 month before event

# Geometry constants
PIXEL_SIZE_M = 10.0  # Pixel size in meters for bbox calculations
TASK_GEOM_BUFFER_M = 20  # Buffer size in meters for task geometry
MIN_BBOX_PIX10M = 1  # Minimum bbox size in 10m pixels to keep
BBOX_AREA_TO_PIX10M_FACTOR = 100.0  # Conversion: m² to 10m pixels (10m * 10m = 100 m²)

# Sampling constants
RANDOM_SEED = 42  # Random seed for reproducible sampling

# Parallel processing constants
DEFAULT_N_WORKERS = multiprocessing.cpu_count()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-zip",
        type=Path,
        default=DEFAULT_DATA_DIR / "inventories.zip",
        help="Path to the inventories.zip file. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Name of the GeoJSON file to create. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory for output. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--balance-multiplier",
        type=float,
        default=2.5,
        help="Multiplier for location cap (median * multiplier). Defaults to %(default)s.",
    )
    parser.add_argument(
        "--min-box-size-pix",
        type=int,
        default=128,
        help="Minimum task geometry size in 10m pixels. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help="Number of worker processes for parallel CRS computation. Defaults to %(default)s.",
    )
    return parser.parse_args()


def _compute_crs_worker(args: tuple[float, float]) -> str:
    """Worker function for parallel CRS computation."""
    x, y = args
    return get_utm_ups_crs(x, y)


def compute_crs_parallel(
    centroids: Any, n_workers: int = DEFAULT_N_WORKERS
) -> list[str]:
    """Compute UTM/UPS CRS for multiple centroids in parallel.

    Args:
        centroids: GeoSeries of Point geometries.
        n_workers: Number of worker processes to use.

    Returns:
        List of CRS strings, one per centroid.
    """
    # Extract x, y coordinates from centroids (GeoSeries of Points)
    coords = [(cent.x, cent.y) for cent in centroids]

    if n_workers <= 1 or len(coords) == 0:
        # Sequential computation for small datasets or single worker
        crs_list = [get_utm_ups_crs(x, y) for x, y in coords]
    else:
        # Parallel computation
        with multiprocessing.Pool(n_workers) as pool:
            crs_list = list(
                tqdm(
                    pool.imap(_compute_crs_worker, coords),
                    desc="Computing CRS",
                    total=len(coords),
                )
            )

    # Ensure all CRS values are strings (convert CRS objects to strings if needed)
    # If CRS objects, use to_string() method; otherwise str() is fine
    result = []
    for crs in crs_list:
        if hasattr(crs, "to_string"):
            result.append(crs.to_string())
        elif hasattr(crs, "to_epsg"):
            # Convert to EPSG code string
            epsg_code = crs.to_epsg()
            result.append(f"EPSG:{epsg_code}" if epsg_code else str(crs))
        else:
            result.append(str(crs))
    return result


def load_landslide_inventory(zip_path: Path) -> gpd.GeoDataFrame:
    """Load and filter landslide inventory from zip file.

    Args:
        zip_path: Path to the inventories.zip file.

    Returns:
        GeoDataFrame with filtered landslide polygons (event_conf == 1.0).
    """
    logging.info("Loading landslide inventory from %s", zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    # Read shapefile directly from zip using zip:// protocol
    # Format: zip://path/to/file.zip!shapefile.shp
    zip_url = f"zip://{zip_path.absolute()}!{SHAPEFILE_NAME}"
    gdf = gpd.read_file(zip_url)

    # Validate and set CRS
    if gdf.crs is None:
        logging.warning("Input data has no CRS, assuming EPSG:4326")
        gdf = gdf.set_crs(GEO_CRS)
    elif gdf.crs.to_string().lower() != GEO_CRS.lower():
        logging.info("Reprojecting input to %s", GEO_CRS)
        gdf = gdf.to_crs(GEO_CRS)

    # Filter to high confidence events
    if COL_EVENT_CONF not in gdf.columns:
        raise ValueError(f"{COL_EVENT_CONF} column not found in inventory data")
    initial_count = len(gdf)
    gdf = gdf.loc[gdf[COL_EVENT_CONF] == MIN_EVENT_CONFIDENCE].copy()
    logging.info(
        "Filtered to %d high-confidence events (from %d total, %.1f%%)",
        len(gdf),
        initial_count,
        100 * len(gdf) / initial_count if initial_count > 0 else 0,
    )

    # Parse event_date
    if COL_EVENT_DATE not in gdf.columns:
        raise ValueError(f"{COL_EVENT_DATE} column not found in inventory data")
    gdf[COL_EVENT_DATE] = pd.to_datetime(gdf[COL_EVENT_DATE], errors="coerce")

    # Filter to events with valid dates
    date_filtered = gdf[COL_EVENT_DATE].notna()
    gdf = gdf.loc[date_filtered].copy()
    logging.info(
        "Kept %d events with valid event_date (dropped %d)",
        len(gdf),
        (~date_filtered).sum(),
    )

    # Fix invalid geometries
    gdf["geometry"] = gdf["geometry"].buffer(0)
    invalid_count = (~gdf.geometry.is_valid).sum()
    if invalid_count > 0:
        logging.warning("%d geometries remained invalid after buffer(0)", invalid_count)

    # Label as positive samples
    gdf[COL_LABEL] = LABEL_LANDSLIDE

    logging.info("Loaded %d landslide polygons", len(gdf))
    return gdf


def create_negative_samples(
    landslides: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Create negative samples using same geometry 1 month earlier.

    Args:
        landslides: GeoDataFrame with landslide polygons and event_date.

    Returns:
        GeoDataFrame with negative samples (no_landslide label).
    """
    logging.info("Creating negative samples (1 month before event_date)")
    negatives = landslides.copy()

    # Set times to 1 month before event (exact match, no time window)
    negatives[COL_START_TIME] = negatives[COL_EVENT_DATE] - pd.Timedelta(
        days=NEGATIVE_SAMPLE_OFFSET_DAYS
    )
    negatives[COL_END_TIME] = negatives[COL_START_TIME]  # Exact match
    negatives[COL_LABEL] = LABEL_NO_LANDSLIDE

    logging.info("Created %d negative sample polygons", len(negatives))
    return negatives


def _remove_overlaps_from_negative(
    geom_metric: Any,
    positives_before_metric: gpd.GeoDataFrame,
    positives_before_sindex: Any,
    negative_idx: Any,
) -> tuple[Any, bool]:
    """Remove overlapping portions from a negative sample geometry.

    Args:
        geom_metric: Negative sample geometry in metric CRS.
        original_area: Original area of the geometry.
        positives_before_metric: Positive samples that occurred before, in metric CRS.
        positives_before_sindex: Spatial index for positives_before_metric.
        negative_idx: Index of the negative sample (to exclude corresponding positive).

    Returns:
        Tuple of (processed_geometry, should_skip) where should_skip is True if
        the negative should be entirely removed due to excessive overlap.
    """
    # Find all intersecting landslides that occurred before this negative sample
    candidate_idx = list(
        positives_before_sindex.query(geom_metric, predicate="intersects")
    )

    if not candidate_idx:
        return geom_metric, False

    # Exclude the corresponding positive sample (same index)
    # since negative samples are created from positive samples with same geometry
    other_landslides_idx = [i for i in candidate_idx if i != negative_idx]

    if not other_landslides_idx:
        return geom_metric, False

    intersecting = positives_before_metric.iloc[other_landslides_idx]
    geoms_to_remove = [
        g for g in intersecting["geometry"] if g is not None and not g.is_empty
    ]

    if not geoms_to_remove:
        return geom_metric, False

    removal_geom = unary_union(geoms_to_remove)
    if removal_geom.is_empty:
        return geom_metric, False

    # Remove the overlapping portion but keep the rest
    # If the result is empty, it will be filtered out later
    trimmed_geom = geom_metric.difference(removal_geom)
    return trimmed_geom, False


def _process_negative_sample(
    row: pd.Series,
    idx: Any,
    geom_crs_str: str,
    negatives_crs: str,
    positives_same_crs: gpd.GeoDataFrame,
) -> tuple[Any, Any] | None:
    """Process a single negative sample to remove overlaps.

    Args:
        row: Row from negatives GeoDataFrame.
        idx: Index of the negative sample.
        geom_crs_str: CRS string for this geometry.
        negatives_crs: Original CRS of negatives GeoDataFrame.
        positives_same_crs: Positive samples with the same CRS.

    Returns:
        Tuple of (processed_geometry_wgs84, idx) if the sample should be kept,
        None if it should be skipped.
    """
    geom = row.geometry
    if geom is None or geom.is_empty:
        return None

    # Get the negative sample's time window
    neg_start_time = row[COL_START_TIME]

    # Reproject this geometry to the group's CRS
    geom_metric = (
        gpd.GeoDataFrame(geometry=[geom], crs=negatives_crs)
        .to_crs(geom_crs_str)
        .geometry.iloc[0]
    )

    # Filter positives to only those that occurred before the negative sample
    if COL_EVENT_DATE not in positives_same_crs.columns:
        raise ValueError(
            f"{COL_EVENT_DATE} column not found in positives. "
            "Required for time-based overlap filtering."
        )
    positives_before = positives_same_crs[
        positives_same_crs[COL_EVENT_DATE] < neg_start_time
    ]

    # Remove overlaps if there are positives before this negative sample
    if len(positives_before) > 0:
        positives_before_metric = positives_before.to_crs(geom_crs_str)
        positives_before_sindex = positives_before_metric.sindex

        geom_metric, should_skip = _remove_overlaps_from_negative(
            geom_metric,
            positives_before_metric,
            positives_before_sindex,
            idx,
        )

        if should_skip:
            return None

    # Keep the negative sample if it still exists
    if geom_metric is None or geom_metric.is_empty:
        return None

    # Convert back to original CRS
    geom_wgs84 = (
        gpd.GeoDataFrame(geometry=[geom_metric], crs=geom_crs_str)
        .to_crs(negatives_crs)
        .geometry.iloc[0]
    )
    return geom_wgs84, idx


def remove_overlapping_landslides(
    negatives: gpd.GeoDataFrame,
    positives: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Remove portions of negative samples that overlap with other landslides.

    Each negative sample corresponds to a specific positive sample (same geometry),
    so we exclude that corresponding positive from the overlap check to avoid
    removing the entire negative sample. Only positives that occurred before
    the negative sample's time window are considered for overlap removal.

    Uses per-geometry CRS determined by get_utm_ups_crs for accurate spatial operations.

    Args:
        negatives: GeoDataFrame of negative sample polygons.
        positives: GeoDataFrame of positive (landslide) polygons.

    Returns:
        Trimmed GeoDataFrame with overlapping portions removed.
    """
    logging.info("Removing overlaps from negative samples")

    # Group negatives by CRS to avoid recomputing spatial index for each negative
    # Ensure COL_GEOM_CRS is stored as strings (for grouping and comparison)
    logging.info("Grouping negatives by CRS for efficient processing")
    negatives[COL_GEOM_CRS] = negatives[COL_GEOM_CRS].astype(str)
    positives[COL_GEOM_CRS] = positives[COL_GEOM_CRS].astype(str)

    crs_groups = negatives.groupby(COL_GEOM_CRS, observed=True)
    n_groups = len(crs_groups)
    logging.info("Found %d unique CRS values", n_groups)

    processed_geoms = []
    indices = []

    # Process each CRS group
    for geom_crs_str, neg_group in tqdm(
        crs_groups, total=n_groups, desc="Processing CRS groups"
    ):
        # Filter positives to only those with the same CRS (more efficient)
        positives_same_crs = positives[positives[COL_GEOM_CRS] == geom_crs_str]

        # Process all negatives in this CRS group
        for idx, row in neg_group.iterrows():
            result = _process_negative_sample(
                row, idx, geom_crs_str, negatives.crs, positives_same_crs
            )
            if result is not None:
                geom_wgs84, idx = result
                processed_geoms.append(geom_wgs84)
                indices.append(idx)

    trimmed = negatives.loc[indices].copy()
    trimmed["geometry"] = processed_geoms
    trimmed = trimmed[trimmed.geometry.notnull() & ~trimmed.geometry.is_empty]
    trimmed["geometry"] = trimmed["geometry"].buffer(0)

    logging.info(
        "Kept %d negative samples after overlap removal (removed %d)",
        len(trimmed),
        len(negatives) - len(trimmed),
    )
    return trimmed


def balance_by_location(
    gdf: gpd.GeoDataFrame, multiplier: float = 2.5
) -> gpd.GeoDataFrame:
    """Balance dataset by location using proportional cap strategy.

    Caps each location at median * multiplier, randomly sampling if needed.

    Args:
        gdf: GeoDataFrame with 'location' column.
        multiplier: Multiplier for location cap (default 2.5).

    Returns:
        Balanced GeoDataFrame.
    """
    if COL_LOCATION not in gdf.columns:
        logging.warning(
            f"No '{COL_LOCATION}' column found, skipping geographic balancing"
        )
        return gdf

    logging.info("Applying geographic balancing (proportional cap)")
    location_counts = gdf[COL_LOCATION].value_counts()
    median_count = float(location_counts.median())
    cap = int(median_count * multiplier)

    logging.info(
        "Location statistics: median=%d, cap=%d (%.1fx), locations=%d",
        int(median_count),
        cap,
        multiplier,
        len(location_counts),
    )

    balanced_rows = []
    for location, count in tqdm(
        location_counts.items(), desc="Balancing locations", total=len(location_counts)
    ):
        location_data = gdf.loc[gdf[COL_LOCATION] == location]
        if count > cap:
            # Sample separately for each label to preserve balance
            sampled_rows = []
            for label in location_data[COL_LABEL].unique():
                label_data = location_data[location_data[COL_LABEL] == label]
                label_count = len(label_data)
                # Sample proportionally to maintain label balance
                label_cap = max(1, int(cap * label_count / count))
                if label_count > label_cap:
                    sampled = label_data.sample(n=label_cap, random_state=RANDOM_SEED)
                else:
                    sampled = label_data
                sampled_rows.append(sampled)
            sampled = pd.concat(sampled_rows, ignore_index=True)
            balanced_rows.append(sampled)
            logging.info(
                "  %s: sampled %d from %d (%.1f%%)",
                location,
                len(sampled),
                count,
                100 * len(sampled) / count,
            )
        else:
            # Keep all
            balanced_rows.append(location_data)

    balanced = pd.concat(balanced_rows, ignore_index=True)
    logging.info(
        "Balanced dataset: %d samples (from %d, %.1f%%)",
        len(balanced),
        len(gdf),
        100 * len(balanced) / len(gdf) if len(gdf) > 0 else 0,
    )
    return balanced


def compute_bbox_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add bbox_area_m2, bbox_pix10m, max_side_pix10m, min_side_pix10m.

    Computes metrics using per-geometry CRS from the COL_GEOM_CRS column.

    Args:
        gdf: GeoDataFrame with geometry column in WGS84 (EPSG:4326) and COL_GEOM_CRS column.

    Returns:
        GeoDataFrame with added bbox metric columns.
    """

    def metrics_for_geom(geom: Any, crs: str) -> tuple[float, int, int]:
        """Calculate area and pixel metrics for a geometry in its appropriate CRS."""
        if geom is None or geom.is_empty:
            return 0.0, 0, 0  # area, max_side_pix10m, min_side_pix10m

        # Reproject to geometry's CRS for accurate area calculation
        geom_gdf = gpd.GeoDataFrame(geometry=[geom], crs=GEO_CRS)
        geom_projected = geom_gdf.to_crs(crs).geometry.iloc[0]

        try:
            env = geom_projected.envelope  # Polygon
            area = float(env.area)
            minx, miny, maxx, maxy = env.bounds
        except GEOSException:
            minx, miny, maxx, maxy = geom_projected.bounds
            area = float(box(minx, miny, maxx, maxy).area)

        width_m = maxx - minx
        height_m = maxy - miny
        # Number of whole 10m pixels that fit along each side
        pix_w = int(math.floor(width_m / PIXEL_SIZE_M))
        pix_h = int(math.floor(height_m / PIXEL_SIZE_M))
        return area, max(pix_w, pix_h), min(pix_w, pix_h)

    out = gdf.copy()

    # Compute metrics for each geometry in its appropriate CRS
    metrics = [
        metrics_for_geom(geom, crs)
        for geom, crs in tqdm(
            zip(out.geometry, out["COL_GEOM_CRS"]),
            desc="Computing bbox metrics",
            total=len(out),
        )
    ]

    out["bbox_area_m2"] = [m[0] for m in metrics]
    out["bbox_pix10m"] = out["bbox_area_m2"] / BBOX_AREA_TO_PIX10M_FACTOR
    out["max_side_pix10m"] = [m[1] for m in metrics]
    out["min_side_pix10m"] = [m[2] for m in metrics]
    return out


def apply_task_geometry_and_filter(
    gdf: gpd.GeoDataFrame,
    min_box_size_pix: int = 128,
) -> gpd.GeoDataFrame:
    """Apply task geometry transformations and filter small polygons.

    Uses per-geometry CRS from the COL_GEOM_CRS column for accurate operations.

    Args:
        gdf: Input GeoDataFrame with 'geometry' column in WGS84, bbox metrics, and COL_GEOM_CRS column.
        min_box_size_pix: Minimum box size in 10m pixels (default: 128).

    Returns:
        GeoDataFrame with task_geom column (as WKT in EPSG:4326).
    """
    out = gdf.copy()

    # Filter out empty or invalid geometries
    valid_mask = out.geometry.notna() & ~out.geometry.is_empty & out.geometry.is_valid
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logging.warning("Dropping %d rows with empty or invalid geometries", n_invalid)
        out = out.loc[valid_mask].copy()

    if len(out) == 0:
        raise ValueError("No valid geometries remaining after filtering")

    # Compute task geometry for each feature using its appropriate CRS
    task_geoms = []
    for idx, row in tqdm(
        out.iterrows(), total=len(out), desc="Creating task geometries"
    ):
        geom = row.geometry
        geom_crs = row[COL_GEOM_CRS]

        # Reproject to geometry's CRS, buffer, then take envelope
        geom_gdf = gpd.GeoDataFrame(geometry=[geom], crs=GEO_CRS)
        geom_projected = geom_gdf.to_crs(geom_crs).geometry.iloc[0]
        task_geom_projected = geom_projected.buffer(TASK_GEOM_BUFFER_M).envelope

        # Convert back to WGS84
        task_geom_wgs84 = (
            gpd.GeoDataFrame(geometry=[task_geom_projected], crs=geom_crs)
            .to_crs(GEO_CRS)
            .geometry.iloc[0]
        )
        task_geoms.append(task_geom_wgs84)

    out["task_geom"] = task_geoms

    # Drop polygons with bbox_pix10m < MIN_BBOX_PIX10M
    n_before = len(out)
    drop_mask = out["bbox_pix10m"] < MIN_BBOX_PIX10M
    n_drop = int(drop_mask.sum())
    pct_drop = (100.0 * n_drop / n_before) if n_before else 0.0

    if n_drop > 0:
        dropped = out.loc[drop_mask]
        label_dist = dropped[COL_LABEL].value_counts().to_dict()
        logging.info(
            "Dropped %d/%d (%.2f%%) polygons with bbox_pix10m < %d: %s",
            n_drop,
            n_before,
            pct_drop,
            MIN_BBOX_PIX10M,
            label_dist,
        )

    out = out.loc[~drop_mask].copy()

    # For small boxes, grow so shortest side fits min_box_size_pix
    small_mask = out["min_side_pix10m"] <= min_box_size_pix
    logging.info(
        "Growing task geometry for %d polygons with min side <= %d pixels",
        small_mask.sum(),
        min_box_size_pix,
    )
    if small_mask.any():
        small_indices = out.loc[small_mask].index
        for idx in tqdm(
            small_indices, desc="Growing small geometries", total=len(small_indices)
        ):
            row = out.loc[idx]
            geom = row["geometry"]
            geom_crs = row[COL_GEOM_CRS]
            missing = min_box_size_pix - row["min_side_pix10m"]
            if missing > 0:
                # Reproject, buffer, convert back (reusing stored CRS)
                geom_gdf = gpd.GeoDataFrame(geometry=[geom], crs=GEO_CRS)
                geom_projected = geom_gdf.to_crs(geom_crs).geometry.iloc[0]
                ceil_half_pix = math.ceil(missing / 2.0)
                grow_m = (ceil_half_pix + 2) * PIXEL_SIZE_M
                grown_geom = geom_projected.buffer(grow_m)
                grown_wgs84 = (
                    gpd.GeoDataFrame(geometry=[grown_geom], crs=geom_crs)
                    .to_crs(GEO_CRS)
                    .geometry.iloc[0]
                )
                out.at[idx, "task_geom"] = grown_wgs84

    # Ensure geometries are valid
    task_geom_series = gpd.GeoSeries(out["task_geom"], crs=GEO_CRS)
    out["task_geom"] = task_geom_series.buffer(0)

    # Reproject to WGS84 and convert to WKT
    tmp = gpd.GeoSeries(out["task_geom"], crs=out.crs)
    task_bbox_wgs84 = tmp.to_crs(GEO_CRS).envelope

    # Validate all are Polygon
    invalid_geoms = task_bbox_wgs84[task_bbox_wgs84.geom_type != "Polygon"]
    if len(invalid_geoms) > 0:
        invalid_types = invalid_geoms.geom_type.value_counts()
        invalid_indices = invalid_geoms.index.tolist()
        raise ValueError(
            f"Found {len(invalid_geoms)} invalid task geometries that are not Polygon:\n"
            f"Geometry types: {dict(invalid_types)}\n"
            f"Row indices: {invalid_indices[:10]}{'...' if len(invalid_indices) > 10 else ''}\n"
            "This should not happen after filtering. Please check input data."
        )

    # Convert to WKT
    out["task_geom"] = task_bbox_wgs84.to_wkt()

    return out


def format_time_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Format time features for positive and negative samples.

    For positive samples: start_time = event_date, end_time = event_date (exact match).
    For negative samples: already set in create_negative_samples().

    Args:
        gdf: GeoDataFrame with event_date and label columns.

    Returns:
        GeoDataFrame with start_time and end_time formatted.
    """
    logging.info("Formatting time features")
    combined = gdf.copy()

    # For positive samples, set times to exactly match event_date
    positive_mask = combined[COL_LABEL] == LABEL_LANDSLIDE
    if positive_mask.any():
        combined.loc[positive_mask, COL_START_TIME] = combined.loc[
            positive_mask, COL_EVENT_DATE
        ]
        combined.loc[positive_mask, COL_END_TIME] = combined.loc[
            positive_mask, COL_EVENT_DATE
        ]  # Exact match, no time window

    # Negative samples should already have start_time and end_time set
    # But ensure they're datetime objects
    combined[COL_START_TIME] = pd.to_datetime(combined[COL_START_TIME], errors="coerce")
    combined[COL_END_TIME] = pd.to_datetime(combined[COL_END_TIME], errors="coerce")

    # Filter out rows with missing times
    initial_count = len(combined)
    time_mask = combined[COL_START_TIME].notna() & combined[COL_END_TIME].notna()
    filtered_count = initial_count - time_mask.sum()
    if filtered_count > 0:
        # Log what labels were filtered
        filtered = combined.loc[~time_mask]
        logging.info(
            "Filtered out %d row(s) with missing time values (%.1f%% of total): %s",
            filtered_count,
            100 * filtered_count / initial_count,
            filtered[COL_LABEL].value_counts().to_dict(),
        )
    combined = combined.loc[time_mask].copy()

    # Validate time ranges
    if len(combined) > 0 and (combined[COL_END_TIME] < combined[COL_START_TIME]).any():
        raise ValueError("Found end_time earlier than start_time.")

    # Add polygon_id
    combined[COL_POLYGON_ID] = combined.index.astype(str)

    return combined


def save_output(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Save the GeoDataFrame to GeoJSON format.

    Args:
        gdf: GeoDataFrame to save.
        output_path: Path for output GeoJSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving GeoJSON to %s", output_path)
    gdf.to_file(output_path, driver="GeoJSON")


def main() -> None:
    """Main entry point for generating landslide label polygons."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load and filter inventory
    positives = load_landslide_inventory(args.input_zip)

    # Compute CRS for positives (will be reused for negatives since they have same geometries)
    logging.info("Computing UTM/UPS CRS for positive samples")
    # Use representative_point() to avoid geographic CRS warning
    centroids = positives.geometry.representative_point()
    positives["COL_GEOM_CRS"] = compute_crs_parallel(centroids, args.workers)

    # Create negative samples (CRS column will be copied automatically)
    negatives = create_negative_samples(positives)

    # Remove overlaps from negatives
    negatives = remove_overlapping_landslides(negatives, positives)

    # Combine positive and negative samples
    combined = pd.concat([positives, negatives], ignore_index=True)

    # Apply geographic balancing
    combined = balance_by_location(combined, args.balance_multiplier)
    logging.info(
        "After geographic balancing - Label distribution: %s",
        combined[COL_LABEL].value_counts().to_dict(),
    )

    # Format time features
    combined = format_time_features(combined)
    logging.info(
        "After time formatting - Label distribution: %s",
        combined[COL_LABEL].value_counts().to_dict(),
    )

    # Compute bbox metrics and task geometries (using per-geometry CRS)
    logging.info("Computing bbox metrics and task geometries")
    combined = compute_bbox_metrics(combined)
    combined = apply_task_geometry_and_filter(combined, args.min_box_size_pix)
    logging.info(
        "After bbox/task geometry filtering - Label distribution: %s",
        combined[COL_LABEL].value_counts().to_dict(),
    )

    # Save output
    output_path = args.data_dir / args.output
    save_output(combined, output_path)
    logging.info("Final dataset shape: %s", combined.shape)
    logging.info("Label distribution:")
    print(combined[COL_LABEL].value_counts().to_string())


if __name__ == "__main__":
    main()
