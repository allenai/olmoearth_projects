"""Gather labels from the WorldCereal RDM."""

import argparse
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
]


def rdm_parquet_to_geojson(parquet_filepath: Path) -> gpd.GeoDataFrame:
    """Sample negatives from WorldCereal RDM files."""
    # which sampling_ewoc_code classes are negatives (basically just excluding maize & unspecified cropland)
    df = gpd.read_parquet(parquet_filepath)
    print(f"Original file length for {parquet_filepath}: {len(df)} instances.")
    df = df[
        # this eliminates 704 (2021) + 2 (2023) points but it could be annual
        # or perennial so its not useful for maize mapping or
        # temporary crop mapping
        (df.sampling_ewoc_code != "cropland_unspecified")
        # this only eliminates 29 points
        & (df.sampling_ewoc_code != "cereals")
    ]
    print(f"After filtering {parquet_filepath}: {len(df)} instances.")
    df.valid_time = pd.to_datetime(df.valid_time)
    # so far, the 2 parquet files were both collected during short rains so
    # we can just take the year
    df["year"] = df.valid_time.dt.year

    return df[["sampling_ewoc_code", "valid_time", "year", "geometry"]]


def collate_parquet_folders(parquet_folder: Path) -> gpd.GeoDataFrame:
    """We use the following files for negative sampling.

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
        assert val in NON_TEMPORARY_CROPS + TEMPORARY_CROPS

    df["is_crop"] = df.apply(lambda x: x.sampling_ewoc_code in TEMPORARY_CROPS, axis=1)
    return df


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
    samples = collate_parquet_folders(parquet_folder=label_dir / "rdm_parquet")
    samples.to_file(label_dir / "labels.geojson", driver="GeoJSON")
