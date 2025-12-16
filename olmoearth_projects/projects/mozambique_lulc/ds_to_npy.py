"""Save the dataset as a npy of pixel timeseries."""

import torch
from rslearn.config import DType
from rslearn.dataset import Dataset
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.segmentation import SegmentationTask
from tqdm import tqdm
from upath import UPath


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


if __name__ == "__main__":
    ds = load_dataset(
        path=UPath(
            "/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251202"
        )
    )

    x, y = [], []
    for i in tqdm(range(len(ds))):
        label = ds[i][1]["segment"]["valid"]
        s2 = ds[i][0]["sentinel2_l2a"]

        # isolate the target pixel
        target_pixels = torch.argwhere(label)
        assert target_pixels.shape[0] == 1
        target_pixel_x, target_pixel_y = target_pixels[0][0], target_pixels[0][1]

        assert label[target_pixel_x, target_pixel_y] != 0  # 0 is missing
        y.append(label[target_pixel_x, target_pixel_y])
        x.append(s2[:, target_pixel_x, target_pixel_y])

    x_np = torch.stack(x, dim=0).numpy()
    y_np = torch.concat(y)
    print(x_np.shape, y_np.shape)
