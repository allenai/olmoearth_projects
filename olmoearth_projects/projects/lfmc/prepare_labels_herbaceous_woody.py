"""Prepare LFMC labels for the herbaceous and woody models from the Globe-LFMC 2.0 dataset.

Example to prepare labels for the herbaceous and woody model for the CONUS region west of 100°W:

```bash
uv run python -m olmoearth_projects.projects.lfmc.prepare_labels_herbaceous_woody \
    --output_dir $(pwd)/olmoearth_run_data/lfmc/ \
    --preset conus \
    --bbox="-124.7844079,24.7433195,-100,49.3457868"
```
"""

import argparse
import tempfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd
import requests
from dateutil import parser as dateutil_parser
from tqdm import tqdm

CONUS_STATES = [
    "Alabama",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]


class Column(StrEnum):
    """Columns in the LFMC CSV file."""

    SORTING_ID = "sorting_id"
    CONTACT = "contact"
    SITE_NAME = "site_name"
    COUNTRY = "country"
    STATE_REGION = "state_region"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    SAMPLING_DATE = "sampling_date"
    PROTOCOL = "protocol"
    VALUE = "value"
    SPECIES_COLLECTED = "species_collected"
    SPECIES_FUNCTIONAL_TYPE = "species_functional_type"


SHEET_NAME = "LFMC data"

COLUMN_MAP = {
    "Sorting ID": Column.SORTING_ID,
    "Contact": Column.CONTACT,
    "Site name": Column.SITE_NAME,
    "Country": Column.COUNTRY,
    "State/Region": Column.STATE_REGION,
    "Latitude (WGS84, EPSG:4326)": Column.LATITUDE,
    "Longitude (WGS84, EPSG:4326)": Column.LONGITUDE,
    "Sampling date (YYYYMMDD)": Column.SAMPLING_DATE,
    "Protocol": Column.PROTOCOL,
    "LFMC value (%)": Column.VALUE,
    "Species collected": Column.SPECIES_COLLECTED,
    "Species functional type": Column.SPECIES_FUNCTIONAL_TYPE,
}

TASK_NAME_COLUMN = "task_name"
START_TIME_COLUMN = "start_time"
END_TIME_COLUMN = "end_time"
FUEL_TYPE_COLUMN = "fuel_type"

HERBACEOUS_FUNCTIONAL_TYPES = frozenset(["forb", "grass"])
WOODY_FUNCTIONAL_TYPES = frozenset(
    ["large shrub", "shrub", "small tree", "subshrub", "tree"]
)

FIGSHARE_ARTICLE_ID = 25413790
FIGSHARE_FILE_ID = 45049786


def get_workbook_download_url() -> str:
    """Get the workbook download URL from the Figshare API."""
    api_url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    files = response.json()
    for f in files:
        if f["id"] == FIGSHARE_FILE_ID:
            return f["download_url"]
    raise ValueError(
        f"File {FIGSHARE_FILE_ID} not found in article {FIGSHARE_ARTICLE_ID}"
    )


def parse_bounding_box(bbox_str: str) -> tuple[float, float, float, float]:
    """Parse bounding box string into coordinates.

    Args:
        bbox_str: Bounding box in format "min_lon,min_lat,max_lon,max_lat"

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)

    Raises:
        ValueError: If the bounding box format is invalid
    """
    try:
        coords = [float(x.strip()) for x in bbox_str.split(",")]
        if len(coords) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")

        min_lon, min_lat, max_lon, max_lat = coords

        # Validate coordinate ranges
        if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if min_lon >= max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if min_lat >= max_lat:
            raise ValueError("min_lat must be less than max_lat")

        return min_lon, min_lat, max_lon, max_lat
    except ValueError as e:
        if "could not convert string to float" in str(e):
            raise ValueError("All coordinates must be valid numbers") from e
        raise


def download_workbook(output_path: Path) -> None:
    """Download the workbook file.

    Args:
        output_path: path to write the workbook file
    """
    download_url = get_workbook_download_url()
    response = requests.get(download_url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as progress_bar:
        with open(output_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")
    if progress_bar.n == 0:
        raise RuntimeError(
            "Downloaded file is empty. The download URL may have changed."
        )


def process_fuel_type_data(
    data_df: pd.DataFrame, fuel_type_filter: str | None = None
) -> pd.DataFrame:
    """Process and group data for a specific fuel type or all data."""
    if fuel_type_filter:
        filtered_df = data_df[data_df[FUEL_TYPE_COLUMN] == fuel_type_filter].copy()
        print(
            f"\nProcessing {fuel_type_filter} samples: {len(filtered_df)} raw samples"
        )
    else:
        filtered_df = data_df.copy()
        print(f"\nProcessing all samples: {len(filtered_df)} raw samples")

    if len(filtered_df) == 0:
        return pd.DataFrame()

    # Group by location, date, and fuel type, then aggregate
    # From the Globe-LFMC-2.0 paper:
    # "For remote sensing applications, it is recommended to average the LFMC measurements taken on
    # the same date and located within the same pixel of the product employed in the study. The
    # choice of which functional type to include in the average can be guided by the land cover type
    # of that pixel. For example, in open canopy forests, both trees and shrubs (or grass) could be
    # included."
    grouped_df = filtered_df.groupby(
        [
            Column.LATITUDE,
            Column.LONGITUDE,
            Column.SAMPLING_DATE,
            FUEL_TYPE_COLUMN,
        ],
        as_index=False,
    ).agg(
        {
            Column.SITE_NAME: "first",
            Column.STATE_REGION: "first",
            Column.COUNTRY: "first",
            Column.VALUE: "mean",
        }
    )

    # Create unique task names by combining site name with count suffix
    site_counts = grouped_df.groupby(Column.SITE_NAME).cumcount() + 1
    grouped_df[TASK_NAME_COLUMN] = (
        grouped_df[Column.SITE_NAME].astype(str)
        + "_"
        + site_counts.astype(str).str.zfill(5)
    )

    # Add time columns
    grouped_df[START_TIME_COLUMN] = pd.to_datetime(
        grouped_df[Column.SAMPLING_DATE], errors="raise"
    )
    grouped_df[END_TIME_COLUMN] = pd.to_datetime(
        grouped_df[Column.SAMPLING_DATE], errors="raise"
    )

    print(f"  Number of tasks: {grouped_df[TASK_NAME_COLUMN].nunique()}")
    print(f"  Number of samples: {len(grouped_df)}")
    print(f"  Min start time: {grouped_df[START_TIME_COLUMN].min()}")
    print(f"  Max end time: {grouped_df[END_TIME_COLUMN].max()}")

    if len(grouped_df) > 0:
        mean_value = grouped_df[Column.VALUE].mean()
        std_value = grouped_df[Column.VALUE].std()
        min_value = grouped_df[Column.VALUE].min()
        max_value = grouped_df[Column.VALUE].max()
        print(f"  LFMC mean: {mean_value:.2f} ± {std_value:.2f}")
        print(f"  LFMC min: {min_value:.2f}")
        print(f"  LFMC max: {max_value:.2f}")

    return grouped_df


def create_csv(
    input_workbook_path: Path,
    output_dir: Path,
    start_date: datetime,
    country_filter: str | None,
    state_region_filter: list[str] | None,
    bounding_box: tuple[float, float, float, float] | None,
) -> None:
    """Create the CSV files.

    Args:
        input_workbook_path: path to the workbook file
        output_dir: path to the output directory
        start_date: start date
        country_filter: country filter
        state_region_filter: state region filter
        bounding_box: bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    print("Reading the workbook")
    df = pd.read_excel(
        input_workbook_path,
        sheet_name=SHEET_NAME,
        usecols=list(COLUMN_MAP.keys()),
        engine="openpyxl",
    )
    df = df.rename(columns=COLUMN_MAP)
    print(f"Initial number of samples: {len(df)}")

    # Calculate 99.9% percentile and clip LFMC values
    percentile_99_9 = round(df[Column.VALUE].quantile(0.999))
    print(
        f"99.9% percentile of LFMC values: {percentile_99_9:.2f} (rounded to: {percentile_99_9})"
    )
    df[Column.VALUE] = df[Column.VALUE].clip(lower=0, upper=percentile_99_9)
    print(f"Clipped LFMC values to range [0, {percentile_99_9}]")

    df = df[df[Column.SAMPLING_DATE] >= start_date]
    print(f"After filtering by date: {len(df)} samples")

    if country_filter is not None:
        df = df[df[Column.COUNTRY] == country_filter]
        print(f"After filtering by country: {len(df)} samples")

    if state_region_filter is not None:
        df = df[df[Column.STATE_REGION].isin(state_region_filter)]
        print(f"After filtering by state/region: {len(df)} samples")

    if bounding_box is not None:
        min_lon, min_lat, max_lon, max_lat = bounding_box
        df = df[
            (df[Column.LONGITUDE] >= min_lon)
            & (df[Column.LONGITUDE] <= max_lon)
            & (df[Column.LATITUDE] >= min_lat)
            & (df[Column.LATITUDE] <= max_lat)
        ]
        print(
            f"After filtering by bounding box [{min_lon}, {min_lat}, {max_lon}, {max_lat}]: {len(df)} samples"
        )

    # Filter out rows with NaN LFMC values
    initial_count = len(df)
    df = df.dropna(subset=[Column.VALUE])
    final_count = len(df)
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} rows with NaN LFMC values")
    print(f"After filtering NaN LFMC values: {len(df)} samples")
    print(df[Column.VALUE].describe())

    # Show unique species functional types before grouping
    unique_functional_types = df[Column.SPECIES_FUNCTIONAL_TYPE].unique()
    print(
        f"Unique species functional types ({len(unique_functional_types)}): {list(unique_functional_types)}"
    )

    # Add fuel type and value columns based on species functional type
    species_type_lower = df[Column.SPECIES_FUNCTIONAL_TYPE].str.lower()

    # Create fuel type column
    df[FUEL_TYPE_COLUMN] = None
    df.loc[species_type_lower.isin(HERBACEOUS_FUNCTIONAL_TYPES), FUEL_TYPE_COLUMN] = (
        "herbaceous"
    )
    df.loc[species_type_lower.isin(WOODY_FUNCTIONAL_TYPES), FUEL_TYPE_COLUMN] = "woody"

    # Filter to only herbaceous and woody samples
    df = df[df[FUEL_TYPE_COLUMN].notna()]
    print(f"After filtering to herbaceous/woody samples: {len(df)} samples")

    # Process all three datasets
    all_df = process_fuel_type_data(df)
    herbaceous_df = process_fuel_type_data(df, "herbaceous")
    woody_df = process_fuel_type_data(df, "woody")

    # Save CSV files
    all_csv_path = output_dir / "labels_all.csv"
    all_df.to_csv(all_csv_path, index=False)
    print(f"\nCreated all samples CSV: {all_csv_path} ({len(all_df)} samples)")

    herbaceous_csv_path = output_dir / "labels_herbaceous.csv"
    herbaceous_df.to_csv(herbaceous_csv_path, index=False)
    print(
        f"Created herbaceous-only CSV: {herbaceous_csv_path} ({len(herbaceous_df)} samples)"
    )

    woody_csv_path = output_dir / "labels_woody.csv"
    woody_df.to_csv(woody_csv_path, index=False)
    print(f"Created woody-only CSV: {woody_csv_path} ({len(woody_df)} samples)")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser("Creates the LFMC CSV files")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="Path to the output directory for CSV files",
    )
    parser.add_argument(
        "--start_date",
        type=dateutil_parser.parse,
        default=datetime(2015, 7, 5),  # Earliest date with all modalities available
        help="Start date to filter the data",
    )
    parser.add_argument(
        "--preset",
        choices=["conus", "global"],
        default="global",
        help="Preset for the country and state/region filter",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box in format 'min_lon,min_lat,max_lon,max_lat' (e.g., '-125,32,-114,42')",
    )
    args = parser.parse_args()

    if args.preset == "global":
        country_filter = None
        state_region_filter = None
    elif args.preset == "conus":
        country_filter = "USA"
        state_region_filter = CONUS_STATES
    else:
        raise ValueError(f"Invalid preset: {args.preset}")

    # Parse bounding box if provided
    bounding_box = None
    if args.bbox is not None:
        bounding_box = parse_bounding_box(args.bbox)

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    with tempfile.TemporaryDirectory() as temp_dir:
        workbook_path = Path(temp_dir) / "lfmc.xlsx"
        download_workbook(workbook_path)
        create_csv(
            workbook_path,
            output_dir,
            args.start_date,
            country_filter,
            state_region_filter,
            bounding_box,
        )


if __name__ == "__main__":
    main()
