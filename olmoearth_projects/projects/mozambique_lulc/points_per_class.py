"""Count how many classes are in each split."""

import argparse

from rslearn.dataset.dataset import Dataset
from upath import UPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(show_progress=True)
    output_dict: dict[str, dict[str, int]] = {}
    for window in windows:
        split = window.options["split"]
        category = window.options["category"]

        if category not in output_dict:
            output_dict[category] = {"train": 0, "val": 0, "test": 0}

        output_dict[category][split] += 1

    print(output_dict)
