"""We know where the maize is growing.

We need to figure out where the maize is not growing.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def rdm_parquet_to_geojson(parquet_filepath: Path) -> gpd.GeoDataFrame:
    """Sample negatives from WorldCereal RDM files."""
    # which sampling_ewoc_code classes are negatives (basically just excluding maize)
    df = gpd.read_parquet(parquet_filepath)
    print(f"Original file length for {parquet_filepath}: {len(df)} instances.")
    df = df[df.sampling_ewoc_code != "maize"]
    print(f"After filtering {parquet_filepath}: {len(df)} instances.")
    df.valid_time = pd.to_datetime(df.valid_time)
    df["year"] = df.valid_time.dt.year

    return df[["sampling_ewoc_code", "valid_time", "year", "geometry"]]


def collate_negatives(parquet_folder: Path) -> gpd.GeoDataFrame:
    """We use the following files for negative sampling.

    1. https://rdm.esa-worldcereal.org/collections/2018_sen_jecamcirad_poly_111
    2. https://rdm.esa-worldcereal.org/collections/2019_sen_jecamcirad_poly_111
    3. https://rdm.esa-worldcereal.org/collections/2019_af_dewatrain1_poly_100
    4. https://rdm.esa-worldcereal.org/collections/2022_glo_ewocval_poly_111
    5. https://rdm.esa-worldcereal.org/collections/2021_gha_ctsurveygeoglam_poly_110
    6. https://rdm.esa-worldcereal.org/collections/2021_glo_ewocval_poly_111
    """
    dfs = []
    for filename in [
        "2022_glo_ewocval_poly_111_dataset.parquet",
        "2018_sen_jecamcirad_poly_111_dataset.parquet",
        "2019_sen_jecamcirad_poly_111_dataset.parquet",
        "2021_gha_ctsurveygeoglam_poly_110_dataset.parquet",
        "2019_af_dewatrain1_poly_100_dataset.parquet",
        "2021_glo_ewocval_poly_111_dataset.parquet",
    ]:
        f_df = rdm_parquet_to_geojson(parquet_folder / filename)
        f_df["filename"] = filename
        f_df["unique_field_id"] = "n/a"
        dfs.append(f_df)
    df = pd.concat(dfs)
    df["crop"] = "not_maize"
    return df


def load_positives(geojson_file: Path) -> gpd.GeoDataFrame:
    """Load positive points, ensure the keys match the negative points."""
    df = gpd.read_file(geojson_file)
    df["valid_time"] = pd.to_datetime(df.end_time)
    df["year"] = df["valid_time"].dt.year
    df["sampling_ewoc_code"] = df["crop"]  # should all be maize
    df["filename"] = geojson_file.name
    return df[
        [
            "sampling_ewoc_code",
            "valid_time",
            "year",
            "geometry",
            "crop",
            "unique_field_id",
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
    negatives = collate_negatives(
        parquet_folder=label_dir / "nigeria_maize_negative_sampling"
    )
    positives = load_positives(
        geojson_file=label_dir / "Clean maize data in Nigeria_buffered.geojson"
    )
    combined = pd.concat([positives, negatives])
    combined.to_file(label_dir / "combined_labels.geojson", driver="GeoJSON")
