"""Check the label quality of the dataset."""

import argparse

import geopandas as gpd
from rslearn.dataset.dataset import Dataset
from rslearn.utils.geometry import WGS84_PROJECTION
from upath import UPath

from olmoearth_projects.utils.label_quality import check_label_quality

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to assess. Use 'all' to assess all splits",
    )
    parser.add_argument("--crop_type", action="store_true", default=False)

    args = parser.parse_args()

    if args.split == "all":
        splits_to_keep = ["train", "val", "test"]
    else:
        splits_to_keep = [args.split]

    dataset = Dataset(UPath(args.ds_path))
    if args.crop_type:
        groups = ["crop_type"]
    else:
        groups = ["gaza", "manica", "zambezia"]
    windows = dataset.load_windows(show_progress=True, groups=groups)
    labels, geometry = [], []
    for window in windows:
        split = window.options["split"]
        if split in splits_to_keep:
            labels.append(window.options["category"])
            geometry.append(window.get_geometry().to_projection(WGS84_PROJECTION).shp)

    df = gpd.GeoDataFrame({"label": labels, "geometry": geometry})
    print(f"Checking label quality for {len(df)} instances.", flush=True)
    check_label_quality(df)
