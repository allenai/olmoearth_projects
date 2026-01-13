"""Gather labels from the WorldCereal RDM."""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def rdm_parquet_to_geojson(parquet_filepath: Path) -> gpd.GeoDataFrame:
    """Sample negatives from WorldCereal RDM files."""
    # which sampling_ewoc_code classes are negatives (basically just excluding maize & unspecified cropland)
    df = gpd.read_parquet(parquet_filepath)
    print(f"Original file length for {parquet_filepath}: {len(df)} instances.")
    df = df[
        (df.sampling_ewoc_code != "cropland_unspecified")
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
    """
    dfs = []
    for filename in [
        "2019_ken_nhicropharvest_point_100_dataset.parquet",
        "2021_ken_copernicusgeoglamsr_point_111_dataset.parquet",
    ]:
        f_df = rdm_parquet_to_geojson(parquet_folder / filename)
        f_df["filename"] = filename
        dfs.append(f_df)
    df = pd.concat(dfs)
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
