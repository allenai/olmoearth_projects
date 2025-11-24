"""We know where the maize is growing.

We need to figure out where the maize is not growing.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def rdm_parquet_to_geojson(parquet_filepath: Path) -> gpd.GeoDataFrame:
    """Sample negatives from WorldCereal RDM files."""
    # which sampling_ewoc_code classes are negatives (basically just excluding maize)
    df = gpd.read_parquet(parquet_filepath)
    df = df[df.sampling_ewoc_code != "maize"]
    df["year"] = pd.to_datetime(df.valid_time).dt.year

    return df[["sampling_ewoc_code", "valid_time", "year", "geometry"]]


def collate_negatives(parquet_folder: Path) -> gpd.GeoDataFrame:
    """We use the following files for negative sampling.

    1. https://rdm.esa-worldcereal.org/collections/2018_sen_jecamcirad_poly_111
    2. https://rdm.esa-worldcereal.org/collections/2019_sen_jecamcirad_poly_111
    3. https://rdm.esa-worldcereal.org/collections/2019_af_dewatrain1_poly_100
    4. https://rdm.esa-worldcereal.org/collections/2022_glo_ewocval_poly_111
    5. https://rdm.esa-worldcereal.org/collections/2021_gha_ctsurveygeoglam_poly_110
    """
    dfs = []
    for filename in [
        "2022_glo_ewocval_poly_111_dataset.parquet",
        "2018_sen_jecamcirad_poly_111_dataset.parquet",
        "2019_sen_jecamcirad_poly_111_dataset.parquet",
        "2021_gha_ctsurveygeoglam_poly_110_dataset.parquet",
        "2019_af_dewatrain1_poly_100_dataset.parquet",
    ]:
        dfs.append(rdm_parquet_to_geojson(parquet_folder / filename))
    df = pd.concat(dfs)
    df.to_file("concatenated_negatives.geojson", driver="GeoJSON")


if __name__ == "__main__":
    collate_negatives(parquet_folder=Path("nigeria_maize_negative_sampling"))
