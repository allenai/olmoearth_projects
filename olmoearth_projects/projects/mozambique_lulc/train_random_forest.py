"""Save the dataset as a npy of pixel timeseries."""

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from rslearn.config import DType
from rslearn.dataset import Dataset
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.segmentation import SegmentationTask
from tqdm import tqdm
from upath import UPath


@dataclass
class AllData:
    """Handy way to store all the data for training the RF."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_dataset(
    path: UPath, crop_type: bool = False, split: str = "train"
) -> ModelDataset:
    """Load dataset."""
    print("Loading dataset.")
    dataset = ModelDataset(
        dataset=Dataset(path=path),
        # just apply normalization here
        split_config=SplitConfig(
            groups=["crop_type"] if crop_type else ["gaza", "manica", "zambezia"],
            tags={"split": split},
            transforms=[
                OlmoEarthNormalize(
                    band_names={
                        "sentinel2_l2a": [
                            "B02",
                            "B03",
                            "B04",
                            "B08",
                            "B05",
                            "B06",
                            "B07",
                            "B8A",
                            "B11",
                            "B12",
                            "B01",
                            "B09",
                        ]
                    }
                )
            ],
        ),
        inputs={
            "sentinel2_l2a": DataInput(
                data_type="raster",
                layers=["sentinel2"],
                bands=[
                    "B02",
                    "B03",
                    "B04",
                    "B08",
                    "B05",
                    "B06",
                    "B07",
                    "B8A",
                    "B11",
                    "B12",
                    "B01",
                    "B09",
                ],
                passthrough=True,
                load_all_item_groups=True,
                load_all_layers=True,
            ),
            "label": DataInput(
                data_type="raster",
                layers=["label_raster"],
                bands=["label"],
                is_target=True,
                dtype=DType.INT32,
            ),
        },
        task=MultiTask(
            tasks={"segment": SegmentationTask(num_classes=8, zero_is_invalid=True)},
            input_mapping={"segment": {"label": "targets"}},
        ),
        workers=32,
        fix_patch_pick=True,
    )
    return dataset


def dataset_to_npy(ds: ModelDataset) -> tuple[np.ndarray, np.ndarray]:
    """Load an rslearn dataset to npy arrays."""
    x, y = [], []
    for i in tqdm(range(len(ds))):
        label = ds[i][1]["segment"]["classes"]
        s2 = ds[i][0]["sentinel2_l2a"]

        # isolate the target pixel
        target_pixels = torch.argwhere(label)
        assert target_pixels.shape[0] == 1
        target_pixel_x, target_pixel_y = target_pixels[0][0], target_pixels[0][1]

        assert label[target_pixel_x, target_pixel_y] != 0  # 0 is missing
        y.append(label[target_pixel_x, target_pixel_y])
        x.append(s2[:, target_pixel_x, target_pixel_y])

    x_np = torch.stack(x, dim=0).numpy()
    y_np = torch.stack(y, dim=0).numpy()

    return x_np, y_np


def load_npys(ds_path: UPath, npy_path: UPath, crop_type: bool) -> AllData:
    """Load saved npys. If they don't exist, recreate them."""
    alldata_dict: dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        expected_suffix = f"{'_crop_type' if crop_type else ''}_{split}.npy"
        if not (npy_path / f"x{expected_suffix}").exists():
            print(f"Missing npys for {split}, loading from data.")
            ds = load_dataset(ds_path, crop_type, split)
            x, y = dataset_to_npy(ds)
            np.save(npy_path / f"x{'_crop_type' if crop_type else ''}_{split}.npy", x)
            np.save(npy_path / f"y{'_crop_type' if crop_type else ''}_{split}.npy", y)
        else:
            print("Loading existing npys.")
            x = np.load(npy_path / f"x{'_crop_type' if crop_type else ''}_{split}.npy")
            y = np.load(npy_path / f"y{'_crop_type' if crop_type else ''}_{split}.npy")
        alldata_dict[f"x_{split}"] = x
        alldata_dict[f"y_{split}"] = y

    return AllData(**alldata_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        default="/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251202",
        type=str,
    )
    parser.add_argument(
        "--npy_path",
        type=str,
        required=True,
    )
    parser.add_argument("--crop_type", action="store_true", default=False)
    args = parser.parse_args()

    all_data = load_npys(UPath(args.ds_path), UPath(args.npy_path), args.crop_type)
    print(all_data)
