"""Create windows for crop type mapping from GPKG files (fixed splits)."""

import argparse
import multiprocessing
from collections.abc import Iterable
from datetime import UTC, datetime
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
from rslp.utils.windows import calculate_bounds
from upath import UPath

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label"

CLASS_MAP = {
    0: "Water",
    1: "Bare Ground",
    2: "Rangeland",
    3: "Flooded Vegetation",
    4: "Trees",
    5: "Cropland",
    6: "Buildings",
}

# Per-province temporal coverage (UTC)
PROVINCE_TIME = {
    "gaza": (
        datetime(2024, 10, 23, tzinfo=UTC),
        datetime(2025, 5, 7, tzinfo=UTC),
    ),
    "manica": (
        datetime(2024, 11, 23, tzinfo=UTC),
        datetime(2025, 6, 7, tzinfo=UTC),
    ),
    "zambezia": (
        datetime(2024, 11, 23, tzinfo=UTC),
        datetime(2025, 6, 7, tzinfo=UTC),
    ),
}


def process_gpkg(gpkg_path: UPath) -> gpd.GeoDataFrame:
    """Load a GPKG and ensure lon/lat in WGS84; expect 'fid' and 'class' columns."""
    gdf = gpd.read_file(str(gpkg_path))

    # Normalize CRS to WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    required_cols = {"class", "geometry"}
    missing = [c for c in required_cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"{gpkg_path}: missing required column(s): {missing}")

    return gdf


def iter_points(gdf: gpd.GeoDataFrame) -> Iterable[tuple[int, float, float, int]]:
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
        category = int(row["class"])
        yield fid, lat, lon, category


def create_window(
    rec: tuple[int, float, float, int],
    ds_path: UPath,
    group_name: str,
    split: str,
    window_size: int,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Create a single window and write label layer."""
    fid, latitude, longitude, category_id = rec
    category_label = CLASS_MAP.get(category_id, f"Unknown_{category_id}")

    # Geometry/projection
    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    # Group = province name; split is taken from file name (train/test)
    group = group_name
    window_name = f"{fid}_{latitude:.6f}_{longitude:.6f}"

    window = Window(
        path=Window.get_window_root(ds_path, group, window_name),
        group=group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(start_time, end_time),
        options={
            "split": split,  # 'train' or 'test' as provided
            "category_id": category_id,
            "category": category_label,
            "fid": fid,
            "source": "gpkg",
        },
    )

    if split == "train":
        # split into a train and val set using the spatial data
        # splitter, keep the test set as it was originally
        splitter = SpatialDataSplitter(
            train_prop=0.8, val_prop=0.2, test_prop=0.0, grid_size=32
        )
        split = splitter.choose_split_for_window(window)
        window.options["split"] = split
    window.save()

    # Label layer (same as before, using window geometry)
    feature = Feature(
        window.get_geometry(),
        {
            "category_id": category_id,
            "category": category_label,
            "fid": fid,
            "split": split,
        },
    )
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(LABEL_LAYER)


def create_windows_from_gpkg(
    gpkg_path: UPath,
    ds_path: UPath,
    group_name: str,
    split: str,
    window_size: int,
    max_workers: int,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Create windows from a single GPKG file."""
    gdf = process_gpkg(gpkg_path)
    records = list(iter_points(gdf))

    jobs = [
        dict(
            rec=rec,
            ds_path=ds_path,
            group_name=group_name,
            split=split,
            window_size=window_size,
            start_time=start_time,
            end_time=end_time,
        )
        for rec in records
    ]

    print(
        f"[{group_name}:{split}] file={gpkg_path.name} features={len(jobs)} "
        f"time={start_time.date()}→{end_time.date()}"
    )

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
        "--gpkg_dir",
        type=str,
        required=True,
        help="Directory containing gaza_[train|test].gpkg, manica_[train|test].gpkg, zambezia_[train|test].gpkg",
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

    gpkg_dir = Path(args.gpkg_dir)
    ds_path = UPath(args.ds_path)

    expected = [
        ("gaza", "train", gpkg_dir / "gaza_train.gpkg"),
        ("gaza", "test", gpkg_dir / "gaza_test.gpkg"),
        ("manica", "train", gpkg_dir / "manica_train.gpkg"),
        ("manica", "test", gpkg_dir / "manica_test.gpkg"),
        ("zambezia", "train", gpkg_dir / "zambezia_train.gpkg"),
        ("zambezia", "test", gpkg_dir / "zambezia_test.gpkg"),
    ]

    # Basic checks
    for province, _, path in expected:
        if province not in PROVINCE_TIME:
            raise ValueError(f"Unknown province '{province}'")
        if not path.exists():
            raise FileNotFoundError(f"Missing expected file: {path}")

    # Run per file
    for province, split, path in expected:
        start_time, end_time = PROVINCE_TIME[province]
        create_windows_from_gpkg(
            gpkg_path=UPath(path),
            ds_path=ds_path,
            group_name=province,  # group == province
            split=split,  # honor provided split
            window_size=args.window_size,
            max_workers=args.max_workers,
            start_time=start_time,
            end_time=end_time,
        )

    print("Done.")
