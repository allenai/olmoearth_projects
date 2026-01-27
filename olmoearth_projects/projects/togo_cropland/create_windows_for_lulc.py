"""Create windows for crop type mapping from GPKG files (fixed splits)."""

import argparse
import multiprocessing
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
import tqdm
from olmoearth_run.runner.tools.data_splitters.spatial_data_splitter import (
    SpatialDataSplitter,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

WINDOW_RESOLUTION = 10
LULC_LABEL_LAYER = "cropland_label"
GROUP = "togo_cropland"
# from https://arxiv.org/pdf/2006.16866:
# "for our hand-labeled dataset we used observations
# acquired March 2019-March 2020"
# so lets say the relevant season is Feb - Sep, 2019
YEAR = 2019
# day, month. This covers Togo's major growing season
# and contains 8 30-day periods.
START_MONTH, START_DAY = 2, 1  # February 1
END_MONTH, END_DAY = 9, 30  # dec 30


def calculate_bounds(
    geometry: STGeometry, window_size: int
) -> tuple[int, int, int, int]:
    """Calculate the bounds of a window around a geometry.

    Args:
        geometry: the geometry to calculate the bounds of.
        window_size: the size of the window.

    Copied from
    https://github.com/allenai/rslearn_projects/blob/master/rslp/utils/windows.py
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")

    if window_size % 2 == 0:
        bounds = (
            int(geometry.shp.x) - window_size // 2,
            int(geometry.shp.y) - window_size // 2,
            int(geometry.shp.x) + window_size // 2,
            int(geometry.shp.y) + window_size // 2,
        )
    else:
        bounds = (
            int(geometry.shp.x) - window_size // 2,
            int(geometry.shp.y) - window_size // 2 - 1,
            int(geometry.shp.x) + window_size // 2 + 1,
            int(geometry.shp.y) + window_size // 2,
        )

    return bounds


def load_shapefiles(geojson_path: UPath) -> gpd.GeoDataFrame:
    """Load all shapefiles in the dir and ensure lon/lat in WGS84."""
    shapefiles = ["crop_merged_v2", "noncrop_merged_v2", "togo_test_majority"]
    gdfs = []
    for filename in shapefiles:
        filepath = geojson_path / filename
        gdf = gpd.read_file(filepath)

        if gdf.crs != "EPSG:4326":
            raise ValueError(
                f"Incorrect crs {gdf.crs} in {geojson_path}. "
                "Was this file created using `create_training_points.py`?"
            )

        if "noncrop" in filepath.name:
            gdf["is_crop"] = False
        else:
            gdf["is_crop"] = True

        if "test" in filepath.name:
            gdf["is_test"] = True
        else:
            gdf["is_test"] = False
        gdf["filename"] = filepath.name
        gdfs.append(gdf)

    return pd.concat(gdfs)


def iter_points(
    gdf: gpd.GeoDataFrame,
) -> Iterable[tuple[float, float, bool, str, bool]]:
    """Yield (fid, latitude, longitude, category) per feature using centroid for polygons."""
    for fid, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, shapely.Point):
            pt = geom
        else:
            pt = geom.centroid
        lon, lat = float(pt.x), float(pt.y)

        # other metadata which may be useful
        is_crop = row.is_crop
        filename = row.filename
        is_test = row.is_test

        yield lat, lon, is_crop, filename, is_test


def create_window(
    rec: tuple[float, float, bool, str, bool],
    ds_path: UPath,
    window_size: int,
) -> None:
    """Create a single window and write label layer."""
    latitude, longitude, is_crop, filename, is_test = rec

    # Geometry/projection
    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    # this is to handle the n/a, which creates some funky windows.
    fid_for_window_name = filename.split(".")[0]
    window_name = f"{latitude:.6f}_{longitude:.6f}_{fid_for_window_name}"

    window = Window(
        path=Window.get_window_root(ds_path, GROUP, window_name),
        group=GROUP,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(
            datetime(YEAR, START_MONTH, START_DAY, tzinfo=UTC),
            datetime(YEAR, END_MONTH, END_DAY, tzinfo=UTC),
        ),
        options={
            "lulc_category": "crop" if is_crop else "non_crop",
            "source_filename": filename,
        },
    )

    splitter = SpatialDataSplitter(
        train_prop=0.8, val_prop=0.2, test_prop=0, grid_size=128
    )
    if not is_test:
        split = splitter.choose_split_for_window(window)
    else:
        split = "test"
    window.options["split"] = split
    window.save()

    # LULC label layers (same as before, using window geometry)
    feature = Feature(
        window.get_geometry(),
        {
            "category_id": is_crop,
            "category": "crop" if is_crop else "non_crop",
            "split": split,
        },
    )
    layer_dir = window.get_layer_dir(LULC_LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(LULC_LABEL_LAYER)


def create_windows_from_geojson(
    geojson_path: UPath,
    ds_path: UPath,
    window_size: int,
    max_workers: int,
) -> None:
    """Create windows from a single GPKG file."""
    gdf = load_shapefiles(geojson_path)
    records = list(iter_points(gdf))

    jobs = [
        dict(
            rec=rec,
            ds_path=ds_path,
            window_size=window_size,
        )
        for rec in records
    ]

    print(f"[file={geojson_path.name} features={len(jobs)} ")

    if max_workers <= 1:
        for kw in tqdm.tqdm(jobs):
            create_window(**kw)
    else:
        p = multiprocessing.Pool(max_workers)
        outputs = star_imap_unordered(p, create_window, jobs)
        for _ in tqdm.tqdm(outputs, total=len(jobs)):
            pass
        p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)

    parser = argparse.ArgumentParser(description="Create windows from shapefiles")
    parser.add_argument(
        "--shapefile_dir",
        type=str,
        required=True,
        help="Path to shapefiles",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset root",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="Window size (pixels per side in projected grid)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Worker processes (set 1 for single-process)",
    )
    args = parser.parse_args()

    shapefile_dir = Path(args.shapefile_dir)
    ds_path = UPath(args.ds_path)
    # Run per file
    create_windows_from_geojson(
        geojson_path=UPath(shapefile_dir),
        ds_path=ds_path,
        window_size=args.window_size,
        max_workers=args.max_workers,
    )

    print("Done.")
