"""Each file has a different format, so we have a function per file."""

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

# we ignore these labels
# since we can't be sure if they are
# maize
RDM_LABELS_TO_IGNORE = [
    "cropland_unspecified",
    "temporary_crops",
]

# https://www.fao.org/giews/countrybrief/country.jsp?code=eth
# based on this, the growing season is from February to December
# or Janury. We will do February to December to avoid any overlap
# this is 11 30-day periods
START_MONTH, START_DAY = 2, 1
END_MONTH, END_DAY = 12, 30


def prepare_maize_data_csv(label_dir: Path) -> gpd.GeoDataFrame:
    """Maize data for the year 2017/18."""
    df = gpd.read_file(label_dir / "maize_data_with_gps.csv")

    # one of the latitudes is an empty string (""), which can't be transformed
    # into a float
    df = df[df.latitude_dd != ""]
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude_dd, df.latitude_dd),
        crs="EPSG:4326",
    )
    gdf["maize_or_not"] = "maize"
    gdf["year"] = 2017
    return gdf[["geometry", "year", "maize_or_not"]]


def prepare_maize_and_non_maize(label_dir: Path) -> gpd.GeoDataFrame:
    """2022 maize and non maize data."""
    df = gpd.read_file(label_dir / "MaizeandNonMaizeSelectedAOI")
    df["maize_or_not"] = df.Class1
    df["year"] = 2022
    return df[["geometry", "year", "maize_or_not"]]


def prepare_selected_districts(label_dir: Path) -> gpd.GeoDataFrame:
    """Maize and non maize data for 2022."""
    files_to_class = {
        "SelectedMaize2022ESS.shp": "maize",
        "SelectedTeff2022ESS.shp": "non_maize",
        "SelectedWheat2022ESS.shp": "non_maize",
    }

    all_dfs = []
    for filename, classification in files_to_class.items():
        gdf = gpd.read_file(label_dir / "SelectedDistrictsForTestinginAOI" / filename)
        gdf["maize_or_not"] = classification
        gdf["year"] = 2022
        all_dfs.append(gdf[["geometry", "year", "maize_or_not"]])

    return pd.concat(all_dfs)


def prepare_non_crop(label_dir: Path) -> gpd.GeoDataFrame:
    """Non crop data for 2022."""
    gdf = gpd.read_file(label_dir / "NonCropin HighMaize Production Woredas")
    gdf["maize_or_not"] = "non_maize"
    # this might be wrong
    gdf["year"] = 2022
    print(
        f"Adding {len(gdf)} non crop points from 'NonCropin HighMaize Production Woredas'"
    )
    # there are multipoints with repeated points.
    gdf = gdf.explode()
    gdf = gdf.drop_duplicates("geometry")
    return gdf[["geometry", "year", "maize_or_not"]]


def prepare_5crops(label_dir: Path) -> gpd.GeoDataFrame:
    """Prepare 5crops data."""
    all_dfs = []
    for filename in [
        "MaizeHPW_FV_Edited.shp",
        "CheckPeas_Cleaned2022HMPZ.shp",
        "Sorghum_Cleaned2022HMPZ.shp",
        "Teff_Cleaned2022HMPZ.shp",
        "Wheat_Cleaned2022HMPZ.shp",
    ]:
        df = gpd.read_file(
            label_dir / "5CropsCleaned in High Maize Production Woredas" / filename
        )
        df["year"] = 2022
        df["maize_or_not"] = "maize" if "Maize" in filename else "non_maize"
        all_dfs.append(df[["geometry", "year", "maize_or_not"]])
        print(f"Adding {len(all_dfs[-1])} points from {filename}")
    return pd.concat(all_dfs)


def rdm_parquet_to_geojson(parquet_filepath: Path) -> gpd.GeoDataFrame:
    """Sample points from WorldCereal RDM files."""
    # which sampling_ewoc_code classes are negatives (basically just excluding maize & unspecified cropland)
    df = gpd.read_parquet(parquet_filepath)
    print(f"Original file length for {parquet_filepath}: {len(df)} instances.")
    df = df[~df.sampling_ewoc_code.isin(RDM_LABELS_TO_IGNORE)]
    print(f"After filtering {parquet_filepath}: {len(df)} instances.")
    df.valid_time = pd.to_datetime(df.valid_time)
    # so far, the 2 parquet files were both collected during short rains so
    # we can just take the year
    df["year"] = df.valid_time.dt.year
    df["maize_or_not"] = df.apply(
        lambda x: "maize" if x.sampling_ewoc_code == "maize" else "non_maize", axis=1
    )

    df["geometry"] = df.geometry.centroid
    return df[["maize_or_not", "year", "geometry"]]


def prepare_worldcereal_rdm(label_dir: Path) -> gpd.GeoDataFrame:
    """Collate world cereal RDM files."""
    dfs: list[gpd.GeoDataFrame] = []
    for filename in [
        "2018_eth_faowapor1_poly_111_dataset.parquet",
        "2018_eth_faowapor2_poly_111_dataset.parquet",
        "2020_eth_ethct2020_point_110_dataset.parquet",
        "2020_eth_nhicropharvest_poly_100_dataset.parquet",
    ]:
        dfs.append(rdm_parquet_to_geojson(label_dir / f"rdm/{filename}"))

    return pd.concat(dfs)


if __name__ == "__main__":
    label_dir = Path("ethiopia_labels")
    use_ess: bool = True
    use_rdm: bool = True

    if not (use_ess or use_rdm):
        raise ValueError("We need to make labels using either ess or rdm (or both).")
    gdfs: list[gpd.GeoDataFrame] = []
    if use_ess:
        gdfs.append(
            pd.concat(
                [
                    # prepare_maize_data_csv(label_dir),
                    # prepare_maize_and_non_maize(label_dir),
                    prepare_selected_districts(label_dir),
                    prepare_non_crop(label_dir),
                    prepare_5crops(label_dir),
                ]
            )
        )

    if use_rdm:
        gdfs.append(prepare_worldcereal_rdm(label_dir))

    labels: gpd.GeoDataFrame = pd.concat(gdfs)

    # add the start_time and end_time columns
    labels["start_time"] = labels.apply(
        lambda x: datetime(x.year, START_MONTH, START_DAY).strftime("%Y-%m-%d"), axis=1
    )
    labels["end_time"] = labels.apply(
        lambda x: datetime(x.year, END_MONTH, END_DAY).strftime("%Y-%m-%d"), axis=1
    )

    file_prefix = "labels"
    if use_ess:
        file_prefix = f"{file_prefix}_ess"
    if use_rdm:
        file_prefix = f"{file_prefix}_rdm"
    labels.to_file(label_dir / f"{file_prefix}.geojson", driver="GeoJSON")
