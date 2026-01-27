"""Create label_raster from label.

If you run this, you will need to update the config.json for the dataset
to include the following entry:

"label_raster": {
      "band_sets": [
        {
          "bands": [
            "label"
          ],
          "dtype": "int32"
        }
      ],
      "type": "raster"
    },
"""

import argparse
import multiprocessing

import numpy as np
import tqdm
from rslearn.dataset.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

LULC_CLASS_NAMES = ["invalid", "non_crop", "crop"]
PROPERTY_NAME = "category"


def create_label_raster(window: Window) -> None:
    """Create label raster for the given window.

    Actually, we are going to draw two label rasters -
    one for cropland mapping, and one for maize mapping.
    """
    # LULC
    label_dir = window.get_layer_dir("lulc_label")
    features = GeojsonVectorFormat().decode_vector(
        label_dir, window.projection, window.bounds
    )
    class_name = features[0].properties[PROPERTY_NAME]
    class_id = LULC_CLASS_NAMES.index(class_name)

    # Draw the class_id in the middle 1x1 of the raster.
    raster = np.zeros(
        (1, window.bounds[3] - window.bounds[1], window.bounds[2] - window.bounds[0]),
        dtype=np.uint8,
    )
    raster[:, raster.shape[1] // 2, raster.shape[2] // 2] = class_id
    raster_dir = window.get_raster_dir("lulc_label_raster", ["lulc_label"])
    GeotiffRasterFormat().encode_raster(
        raster_dir, window.projection, window.bounds, raster
    )
    window.mark_layer_completed("lulc_label_raster")


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
        default=64,
        help="Number of worker processes to use",
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(workers=args.workers, show_progress=True)
    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(create_label_raster, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows)):
        pass
    p.close()
