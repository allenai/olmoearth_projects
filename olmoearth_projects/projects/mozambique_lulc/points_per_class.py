"""Count how many classes are in each split."""

import argparse

import numpy as np
from rslearn.dataset.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

LULC_CLASS_NAMES = [
    "invalid",
    "Water",
    "Bare Ground",
    "Rangeland",
    "Flooded Vegetation",
    "Trees",
    "Cropland",
    "Buildings",
]
CROPTYPE_CLASS_NAMES = [
    "invalid",
    "corn",
    "cassava",
    "rice",
    "sesame",
    "beans",
    "millet",
    "sorghum",
]
PROPERTY_NAME = "category"
BAND_NAME = "label"


def create_label_raster(window: Window) -> None:
    """Create label raster for the given window."""
    label_dir = window.get_layer_dir("label")
    features = GeojsonVectorFormat().decode_vector(
        label_dir, window.projection, window.bounds
    )
    class_name = features[0].properties[PROPERTY_NAME]
    try:
        class_id = LULC_CLASS_NAMES.index(class_name)
    except ValueError:
        class_id = CROPTYPE_CLASS_NAMES.index(class_name)

    # Draw the class_id in the middle 1x1 of the raster.
    raster = np.zeros(
        (1, window.bounds[3] - window.bounds[1], window.bounds[2] - window.bounds[0]),
        dtype=np.uint8,
    )
    raster[:, raster.shape[1] // 2, raster.shape[2] // 2] = class_id
    raster_dir = window.get_raster_dir("label_raster", [BAND_NAME])
    GeotiffRasterFormat().encode_raster(
        raster_dir, window.projection, window.bounds, raster
    )
    window.mark_layer_completed("label_raster")


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
    windows = dataset.load_windows(workers=args.workers, show_progress=True)
    output_dict: dict[str, dict[str, int]] = {}
    for window in windows:
        split = window.options["split"]
        category = window.options["category"]

        if category not in output_dict:
            output_dict[category] = {"train": 0, "val": 0, "test": 0}

        output_dict[category][split] += 1

    print(output_dict)
