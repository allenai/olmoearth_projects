"""Gather labels from the WorldCereal RDM."""

import argparse
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

# These cover all the unique sampling_ewoc_code values in the parquet files. If the parquet files are updated
# this may need to be updated too.
NON_TEMPORARY_CROPS = [
    "non_cropland_incl_perennial",
    "other_permanent_crops",
    "fruits",
    "permanent_crops",
    "herbaceous_vegetation",
    "shrubland",
    "built_up",
    "trees_mixed",
    "mixed_cropland",
    "grasslands",
    "trees_unspecified",
    "wetlands",
    "open_water",
    "nuts",  # permanent crop
]
TEMPORARY_CROPS = [
    "vegetables_fruits",
    "maize",
    "dry_pulses_legumes",
    "wheat",
    "grass_fodder_crops",
    "potatoes",
    "other_oilseeds",
    "herb_spice_medicinal_crops",
    "root_tuber_crops",
    "not_cultivated_fallow",  # temporary crops according to the legend
    "mixed_arable_crops",
    "sunflower",
    "barley",
    "millet",
    "oats",
    "sorghum",
    "soy_soybeans",
    "flower_crops",
    "cereals",
]


# technically, worldcereal labels this as
# a unique not-temporary-crop class, but this leaves
# a lot of room for confusion (e.g. "corn_with_trees"
# might look a lot like corn for the model depending
# on where exactly the point is)
LABELS_TO_IGNORE = [
    "mixed_cropland",
    # Unknown cropped land, either annual or perennial
    "cropland_unspecified",
]

# we will ignore these labels during
# maize mapping since its unclear
# whether they are maize or not.
MAYBE_MAIZE_MAYBE_NOT = [
    "mixed_arable_crops",  # e.g. "maize_mixed_with_yam" or "vegetables_mixed_with_sunflower"
    "cereals",
]


def rdm_parquet_to_geojson(parquet_filepath: Path) -> gpd.GeoDataFrame:
    """Sample negatives from WorldCereal RDM files."""
    # which sampling_ewoc_code classes are negatives (basically just excluding maize & unspecified cropland)
    df = gpd.read_parquet(parquet_filepath)
    print(f"Original file length for {parquet_filepath}: {len(df)} instances.")
    df = df[~df.sampling_ewoc_code.isin(LABELS_TO_IGNORE)]
    print(f"After filtering {parquet_filepath}: {len(df)} instances.")
    df.valid_time = pd.to_datetime(df.valid_time)
    # so far, the 2 parquet files were both collected during short rains so
    # we can just take the year
    df["year"] = df.valid_time.dt.year

    return df[["sampling_ewoc_code", "valid_time", "year", "geometry"]]


def collate_parquet_folders(parquet_folder: Path) -> gpd.GeoDataFrame:
    """We use the following files to obtain labels.

    1. https://rdm.esa-worldcereal.org/collections/2019_ken_nhicropharvest_point_100
    2. https://rdm.esa-worldcereal.org/collections/2021_ken_copernicusgeoglamsr_point_111
    3. GeoGlam 2023, short rains
    """
    dfs = []
    for filename in [
        "2019_ken_nhicropharvest_point_100_dataset.parquet",
        "2021_ken_copernicusgeoglamsr_point_111_dataset.parquet",
        "2022_KEN_COPERNICUS-GEOGLAM-SR_POINT_111.geoparquet",
    ]:
        f_df = rdm_parquet_to_geojson(parquet_folder / filename)
        f_df["filename"] = filename
        dfs.append(f_df)
    df = pd.concat(dfs)

    # check that our list is complete
    for val in df.sampling_ewoc_code.unique():
        assert val in NON_TEMPORARY_CROPS + TEMPORARY_CROPS, f"unexepected {val}"

    df["is_crop"] = df.apply(lambda x: x.sampling_ewoc_code in TEMPORARY_CROPS, axis=1)

    def is_maize(sampling_ewoc_code: str) -> str:
        if sampling_ewoc_code in MAYBE_MAIZE_MAYBE_NOT:
            return "n/a"
        elif sampling_ewoc_code == "maize":
            return "maize"
        else:
            return "not_maize"

    df["is_maize"] = df.apply(lambda x: is_maize(x.sampling_ewoc_code), axis=1)
    return df


def load_gabi_negatives(geojson_path: Path) -> gpd.GeoDataFrame:
    """These are corrective labels created by Gabriel Tseng.

    We assume there is only a geometry column.

    1. 20260126_gabi_negatives.geojson (created on January 26 2026). Since this was
       collected in early 2026, I will assign the valid year as 2025 so that it covers
       the 2025 short rains.
    """
    dfs = []
    for filename, valid_date in [
        ("20260126_gabi_negatives.geojson", datetime(2025, 12, 15))
    ]:
        df = gpd.read_file(geojson_path / filename)
        df["sampling_ewoc_code"] = "non_cropland_incl_perennial"
        df["is_crop"] = False
        df["is_maize"] = "not_maize"
        df["year"] = valid_date.year
        df["valid_time"] = pd.to_datetime(valid_date)
        df["filename"] = filename
        dfs.append(df)
        print(f"Added {len(df)} negative samples from {filename}")

    df = pd.concat(dfs)
    return df[
        [
            "sampling_ewoc_code",
            "valid_time",
            "year",
            "geometry",
            "is_crop",
            "is_maize",
            "filename",
        ]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Path to the labels",
    )
    args = parser.parse_args()
    label_dir = Path(args.label_dir)
    mixed_samples = collate_parquet_folders(parquet_folder=label_dir / "rdm_parquet")
    negative_samples = load_gabi_negatives(geojson_path=label_dir / "gabi_negatives")
    combined_samples = pd.concat([mixed_samples, negative_samples])
    combined_samples.to_file(label_dir / "labels.geojson", driver="GeoJSON")
