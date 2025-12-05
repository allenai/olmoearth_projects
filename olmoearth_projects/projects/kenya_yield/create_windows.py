"""Create windows for crop type mapping from GPKG files (fixed splits)."""

import argparse
import multiprocessing
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import shapely
import tqdm
from olmoearth_run.runner.tools.data_splitters.data_splitter_interface import (
    DataSplitterInterface,
)
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

    # Normalize CRS to WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def iter_points(
    gdf: gpd.GeoDataFrame,
) -> Iterable[tuple[int, float, float, str, float, str, str]]:
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
        crop_type = row["Crop_Type"]
        yield_mt_ha = row["Yield(Mt/ha)"]
        planting_date = row["Planting_Date"]
        harvest_date = row["Harvest_Date"]
        yield fid, lat, lon, crop_type, yield_mt_ha, planting_date, harvest_date


def create_window(
    rec: tuple[int, float, float, str, float, str, str],
    ds_path: UPath,
    split: str,
    window_size: int,
    splitter: DataSplitterInterface,
) -> None:
    """Create a single window and write label layer."""
    fid, lat, lon, crop_type, yield_mt_ha, planting_date, harvest_date = rec
    # Geometry/projection
    src_point = shapely.Point(lon, lat)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(lon, lat)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    # Group = crop type
    group = crop_type
    window_name = f"{fid}_{lat:.6f}_{lon:.6f}"
    start_time = datetime(*[int(d) for d in planting_date.split("-")])  # type: ignore
    end_time = datetime(*[int(d) for d in harvest_date.split("-")])  # type: ignore
    window = Window(
        path=Window.get_window_root(ds_path, group, window_name),
        group=group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(start_time, end_time),
        options={
            "split": split,  # 'train' or 'test' as provided
            "yield_value": yield_mt_ha,
            "fid": fid,
            "source": "gpkg",
        },
    )

    if split == "train":
        split = splitter.choose_split_for_window(window)
        window.options["split"] = split
    window.save()

    # Label layer (same as before, using window geometry)
    feature = Feature(
        window.get_geometry(),
        {
            "yield_value": yield_mt_ha,
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

    splitter = SpatialDataSplitter(
        train_prop=0.8, val_prop=0.1, test_prop=0.1, grid_size=32
    )

    jobs = [
        dict(
            rec=rec,
            ds_path=ds_path,
            window_size=window_size,
            splitter=splitter,
        )
        for rec in records
    ]

    print(f"[file={gdf.name} features={len(jobs)} ")

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
        "--geojson_path",
        type=str,
        required=True,
        help="Path to the geojson compiled by Pula",
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
    parser.add_argument("--crop_type", action="store_true", default=False)
    args = parser.parse_args()

    geojson_path = Path(args.geojson_path)
    ds_path = UPath(args.ds_path)

    # Run per file
    create_windows_from_geojson(
        geojson_path=UPath(geojson_path),
        ds_path=ds_path,
        window_size=args.window_size,
        max_workers=args.max_workers,
    )

    print("Done.")
