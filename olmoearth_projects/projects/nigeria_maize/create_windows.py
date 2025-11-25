"""Create windows for crop type mapping from GPKG files (fixed splits)."""

import argparse
import multiprocessing
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import geopandas as gpd
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
LABEL_LAYER = "label"
GROUP = "nigeria_maize"

CLASS2INT = {
    "not_maize": 0,
    "maize": 1,
}

# day, month
START_MONTH, START_DAY = 6, 1  # june 1
END_MONTH, END_DAY = 12, 15  # dec 15


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


def load_geojson(geojson_path: UPath) -> gpd.GeoDataFrame:
    """Load a geojson and ensure lon/lat in WGS84."""
    gdf = gpd.read_file(geojson_path)

    required_cols = {
        "sampling_ewoc_code",
        "valid_time",
        "year",
        "geometry",
        "crop",
        "unique_field_id",
        "filename",
    }
    missing = [c for c in required_cols if c not in gdf.columns]
    if missing:
        raise ValueError(
            f"{geojson_path}: missing required column(s): {missing}. "
            "Was this file created using `create_training_points.py`?"
        )

    if gdf.crs != "EPSG:4326":
        raise ValueError(
            f"Incorrect crs {gdf.crs} in {geojson_path}. "
            "Was this file created using `create_training_points.py`?"
        )

    return gdf


def iter_points(
    gdf: gpd.GeoDataFrame,
) -> Iterable[tuple[str, float, float, str, str, str, int]]:
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
        category = row.crop
        fid = row.unique_field_id
        crop_type = row.sampling_ewoc_code
        source_filename = row.filename
        year = row.year

        yield fid, lat, lon, category, crop_type, source_filename, year


def create_window(
    rec: tuple[str, float, float, str, str, str, int],
    ds_path: UPath,
    window_size: int,
) -> None:
    """Create a single window and write label layer."""
    fid, latitude, longitude, category, crop_type, source_filename, year = rec
    category_id = CLASS2INT[category]

    # Geometry/projection
    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    # this is to handle the n/a, which creates some funky windows.
    fid_for_window_name = fid
    if fid == "n/a":
        fid_for_window_name = source_filename.split(".")[0]
    window_name = f"{latitude:.6f}_{longitude:.6f}_{fid_for_window_name}"

    window = Window(
        path=Window.get_window_root(ds_path, GROUP, window_name),
        group=GROUP,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(
            datetime(year, START_MONTH, START_DAY),
            datetime(year, END_MONTH, END_DAY),
        ),
        options={
            "category_id": category_id,
            "category": category,
            "fid": fid,
            "crop_type": crop_type,  # not used when learning
            "source_filename": source_filename,
        },
    )

    splitter = SpatialDataSplitter(
        train_prop=0.7, val_prop=0.15, test_prop=0.15, grid_size=128
    )
    split = splitter.choose_split_for_window(window)
    window.options["split"] = split
    window.save()

    # Label layer (same as before, using window geometry)
    feature = Feature(
        window.get_geometry(),
        {
            "category_id": category_id,
            "category": category,
            "fid": fid,
            "split": split,
        },
    )
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(LABEL_LAYER)


def create_windows_from_geojson(
    geojson_path: UPath,
    ds_path: UPath,
    window_size: int,
    max_workers: int,
) -> None:
    """Create windows from a single GPKG file."""
    gdf = load_geojson(geojson_path)
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

    parser = argparse.ArgumentParser(description="Create windows from GPKG files")
    parser.add_argument(
        "--geojson_file",
        type=str,
        required=True,
        help="Path to geojson to load",
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

    geojson_path = Path(args.geojson_file)
    ds_path = UPath(args.ds_path)
    # Run per file
    create_windows_from_geojson(
        geojson_path=UPath(geojson_path),
        ds_path=ds_path,
        window_size=args.window_size,
        max_workers=args.max_workers,
    )

    print("Done.")
