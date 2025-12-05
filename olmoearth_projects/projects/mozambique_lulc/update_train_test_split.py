"""Update splits for dataset."""

import argparse
import multiprocessing

import tqdm
from olmoearth_run.runner.tools.data_splitters.data_splitter_interface import (
    DataSplitterInterface,
)
from olmoearth_run.runner.tools.data_splitters.random_data_splitter import (
    RandomDataSplitter,
)
from olmoearth_run.runner.tools.data_splitters.spatial_data_splitter import (
    SpatialDataSplitter,
)
from rslearn.dataset.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath


def update_train_val_split(window: Window, splitter: DataSplitterInterface) -> None:
    """Create label raster for the given window."""
    if window.options["split"] in ["train", "val"]:
        split = splitter.choose_split_for_window(window)
        print(f"Window was {window.options['split']}, changing to {split}")
        window.options["split"] = split
        window.save()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use",
    )
    parser.add_argument(
        "--splitter",
        type=str,
        default="spatial",
        help="Data splitter to use",
    )
    parser.add_argument("--crop_type", action="store_true", default=False)
    args = parser.parse_args()

    if args.splitter == "spatial":
        splitter = SpatialDataSplitter(
            train_prop=0.9, val_prop=0.1, test_prop=0.0, grid_size=32
        )
    elif args.splitter == "random":
        splitter = RandomDataSplitter(train_prop=0.9, val_prop=0.1, test_prop=0.0)
    else:
        raise ValueError(
            f"Unrecognized splitter {args.splitter}. "
            "Expected one of {random, spatial}."
        )

    dataset = Dataset(UPath(args.ds_path))
    if args.crop_type:
        groups = ["crop_type"]
    else:
        groups = ["gaza", "manica", "zambezia"]
    windows = dataset.load_windows(
        workers=args.workers, show_progress=True, groups=groups
    )

    if args.workers <= 1:
        for window in tqdm.tqdm(windows):
            update_train_val_split(window, splitter)
    else:
        jobs = [dict(splitter=splitter, window=w) for w in windows]
        p = multiprocessing.Pool(args.workers)
        outputs = star_imap_unordered(p, update_train_val_split, jobs)
        for _ in tqdm.tqdm(outputs, total=len(jobs)):
            pass
        p.close()
