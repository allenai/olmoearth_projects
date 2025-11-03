#!/usr/bin/env python3
"""Generate burnt and unburnt CAL FIRE label polygons for OlmoEarth fine-tuning.

This script downloads fire perimeter data from CAL FIRE's Fire and Resource
Assessment Program (FRAP) and processes it to create training labels for
burn scar detection models. It generates:

1. Burnt area polygons from historical fire perimeters
2. Unburnt area polygons as rings around fire perimeters

The script filters fires by year, creates buffer zones around burn perimeters,
and removes overlaps where unburnt rings intersect with recent fires to ensure
label accuracy.

Example:
    $ python Calfire_data_prep.py --start-year 2020 --ring-width 1000 --gap-width 150
"""

import argparse
import logging
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

DEFAULT_DATA_DIR = Path("/weka/dfive-default/hadriens/datasets")
DEFAULT_DOWNLOAD_URL = (
    "https://34c031f8-c9fd-4018-8c5a-4159cdff6b0d-cdn-endpoint.azureedge.net/"
    "-/media/calfire-website/what-we-do/fire-resource-assessment-program---frap/"
    "gis-data/2025/fire241gdb.ashx"
)
DEFAULT_OUTPUT_NAME = "Calfire_2020-2025.json"
FIRE_LAYER_NAME = "firep24_1"
PROJECTED_CRS = "EPSG:3310"
GEO_CRS = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Root directory that holds downloads/ and label_data/ subfolders. "
            "Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Name of the GeoJSON file to create inside label_data/.",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=DEFAULT_DOWNLOAD_URL,
        help="URL of the CAL FIRE geodatabase archive.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Keep fires with YEAR_ greater than or equal to this value.",
    )
    parser.add_argument(
        "--ring-width",
        type=int,
        default=1000,
        help="Width of unburnt rings in meters.",
    )
    parser.add_argument(
        "--gap-width",
        type=int,
        default=150,
        help="Gap between burnt perimeter and unburnt ring in meters.",
    )
    return parser.parse_args()


def download_archive(download_url: str, download_dir: Path) -> Path:
    """Download the CAL FIRE archive if not already present."""
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / "fire241gdb.zip"

    if not archive_path.exists():
        logging.info("Downloading CAL FIRE archive to %s", archive_path)
        try:
            # Create a request with proper headers to avoid 403 errors
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            request = Request(download_url, headers=headers)

            # Download the file
            with urlopen(request) as response, open(archive_path, "wb") as out_file:
                out_file.write(response.read())
        except Exception as e:
            raise RuntimeError(
                f"Failed to download archive from {download_url}: {e}"
            ) from e
    else:
        logging.info("Reusing existing archive at %s", archive_path)

    return archive_path


def extract_archive(archive_path: Path, label_dir: Path) -> None:
    """Extract the geodatabase archive to the label directory."""
    label_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        logging.info("Extracting archive to %s", label_dir)
        zf.extractall(path=label_dir)


def ensure_data_present(
    data_dir: Path, download_url: str, gdb_name: str = "fire24_1.gdb"
) -> Path:
    """Ensure the geodatabase is present, downloading if necessary."""
    label_dir = data_dir / "label_data"
    gdb_path = label_dir / gdb_name

    if gdb_path.exists():
        logging.info("Found geodatabase at %s", gdb_path)
        return gdb_path

    logging.info("Geodatabase not found, downloading fresh copy.")
    download_dir = data_dir / "downloads"
    archive_path = download_archive(download_url, download_dir)
    extract_archive(archive_path, label_dir)

    if not gdb_path.exists():
        raise FileNotFoundError(f"Expected {gdb_path} after extraction.")

    return gdb_path


def load_fire_perimeters(gdb_path: Path, start_year: int) -> gpd.GeoDataFrame:
    """Load and filter fire perimeters from the geodatabase."""
    logging.info("Loading fire perimeters from %s", gdb_path)
    fire_perim = gpd.read_file(gdb_path, layer=FIRE_LAYER_NAME)
    if fire_perim.crs is None:
        raise ValueError("Input data has no CRS defined.")
    if fire_perim.crs.to_string().lower() != GEO_CRS.lower():
        logging.info("Reprojecting input to %s", GEO_CRS)
        fire_perim = fire_perim.to_crs(GEO_CRS)

    fire_perim["ALARM_DATE"] = pd.to_datetime(fire_perim["ALARM_DATE"])
    fire_perim["CONT_DATE"] = pd.to_datetime(fire_perim["CONT_DATE"])

    filtered = fire_perim.loc[
        (fire_perim["YEAR_"] >= start_year) & (fire_perim.CONT_DATE.notnull())
    ].copy()
    filtered["geometry"] = filtered["geometry"].buffer(0)
    if not filtered.geometry.is_valid.all():
        logging.warning("Some geometries remained invalid after buffer(0).")

    filtered["label"] = "burnt"
    filtered["Shape_Area"] = filtered["Shape_Area"].astype("int")
    logging.info("Kept %s fires with YEAR_ >= %s", filtered.shape[0], start_year)
    return filtered


def remove_overlaps_with_recent_fires(
    rings_metric: gpd.GeoDataFrame,
    burnt_metric: gpd.GeoDataFrame,
    gap_days: int = 45,
) -> gpd.GeoDataFrame:
    """Remove portions of unburnt rings that overlap with recent fires.

    Args:
        rings_metric: GeoDataFrame of unburnt ring polygons in projected CRS
        burnt_metric: GeoDataFrame of burnt area polygons in projected CRS
        gap_days: Number of days after containment to consider fires as overlapping
                  (default 45 days allows for fire damage to become visible)

    Returns:
        Trimmed GeoDataFrame with overlapping portions removed
    """
    burnt_sindex = burnt_metric.sindex
    processed_geoms = []
    indices = []

    for idx, row in rings_metric.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        cont_date = row.get("CONT_DATE")
        if pd.isna(cont_date):
            processed_geoms.append(geom)
            indices.append(idx)
            continue

        removal_deadline = cont_date + pd.Timedelta(days=gap_days)
        candidate_idx = list(burnt_sindex.query(geom, predicate="intersects"))

        if candidate_idx:
            relevant = burnt_metric.iloc[candidate_idx]
            mask = relevant["ALARM_DATE"].notna() & (
                relevant["ALARM_DATE"] <= removal_deadline
            )
            if mask.any():
                geoms_to_remove = [
                    g
                    for g in relevant.loc[mask, "geometry"]
                    if g is not None and not g.is_empty
                ]
                if geoms_to_remove:
                    removal_geom = unary_union(geoms_to_remove)
                    if not removal_geom.is_empty:
                        geom = geom.difference(removal_geom)

        if geom is None or geom.is_empty:
            continue

        processed_geoms.append(geom)
        indices.append(idx)

    trimmed = rings_metric.loc[indices].copy()
    trimmed["geometry"] = processed_geoms
    trimmed = trimmed[trimmed.geometry.notnull() & ~trimmed.geometry.is_empty]
    trimmed["geometry"] = trimmed["geometry"].buffer(0)
    return trimmed


def create_unburnt_rings(
    burnt_perimeters: gpd.GeoDataFrame,
    ring_width: float = 500.0,
    gap_width: float = 50.0,
) -> gpd.GeoDataFrame:
    """Create unburnt ring polygons around fire perimeters."""
    logging.info("Generating unburnt rings.")
    metric = burnt_perimeters.to_crs(PROJECTED_CRS)
    metric["ALARM_DATE"] = pd.to_datetime(metric["ALARM_DATE"])
    metric["CONT_DATE"] = pd.to_datetime(metric["CONT_DATE"])

    outer = metric.geometry.buffer(gap_width + ring_width)
    inner = metric.geometry.buffer(gap_width)
    rings_metric = metric.copy()
    rings_metric["geometry"] = outer.difference(inner)

    trimmed = remove_overlaps_with_recent_fires(rings_metric, metric)
    trimmed["label"] = "unburnt"
    trimmed["Shape_Area"] = trimmed.geometry.area.astype("int")

    unburnt = trimmed.to_crs(burnt_perimeters.crs)
    logging.info("Created %s unburnt polygons.", unburnt.shape[0])
    return unburnt


def format_features(combined: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Format feature columns for output.

    Sets start_time to fire alarm date and end_time to 30 days after alarm.
    The 30-day window provides sufficient time for burn scars to be clearly
    visible in satellite imagery while the area is still relatively unchanged.
    """
    logging.info("Formatting feature columns.")
    combined = combined.copy()
    combined["polygon_id"] = combined.index.astype(str)
    combined["start_time"] = pd.to_datetime(combined["CONT_DATE"])
    combined["end_time"] = pd.to_datetime(combined["start_time"]) + pd.Timedelta(
        days=30
    )

    # Filter out rows with missing end_time values
    initial_count = len(combined)
    combined = combined[combined["end_time"].notna()].copy()
    filtered_count = initial_count - len(combined)
    if filtered_count > 0:
        logging.info(
            "Filtered out %d row(s) with missing end_time values (%.1f%% of total)",
            filtered_count,
            100 * filtered_count / initial_count,
        )

    if len(combined) > 0 and (combined["end_time"] < combined["start_time"]).any():
        raise ValueError("Found end_time earlier than start_time.")
    return combined


def save_output(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Save the GeoDataFrame to GeoJSON format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving GeoJSON to %s", output_path)
    gdf.to_file(output_path, driver="GeoJSON")
    gdf.to_file(output_path.with_suffix(".gdb"))


def main() -> None:
    """Main entry point for generating burnt and unburnt label polygons."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gdb_path = ensure_data_present(args.data_dir, args.download_url)
    burnt_perimeters = load_fire_perimeters(gdb_path, args.start_year)
    ring_width = getattr(args, "ring_width", args.__dict__.get("ring-width", 1000))
    gap_width = getattr(args, "gap_width", args.__dict__.get("gap-width", 150))
    unburnt_rings = create_unburnt_rings(burnt_perimeters, ring_width, gap_width)
    combined = pd.concat([burnt_perimeters, unburnt_rings], ignore_index=True)
    combined = format_features(combined)

    output_path = args.data_dir / "label_data" / args.output
    save_output(combined, output_path)
    logging.info("Final dataset shape: %s", combined.shape)


if __name__ == "__main__":
    main()
