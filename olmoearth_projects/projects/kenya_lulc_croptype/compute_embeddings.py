"""Compute OlmoEarth nano embeddings for the central pixel of each window.

Loads satellite imagery windows created by create_windows.py, extracts the
central pixel from each, computes embeddings using the OlmoEarth nano
encoder, and saves the results for downstream outlier analysis.

Usage:
    python compute_embeddings.py \
        --ds_path /path/to/dataset \
        --output_dir /path/to/output
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import tqdm
from olmoearth_pretrain.model_loader import ModelID
from rslearn.config import DType
from rslearn.dataset import Dataset
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.segmentation import SegmentationTask
from upath import UPath

GROUP = "kenya_cropland_maize"
SENTINEL2_BANDS = [
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
CSV_FIELDS = ["window_name", "lulc_label", "crop_type", "split"]


def load_dataset(ds_path: UPath, workers: int) -> ModelDataset:
    """Load the Kenya dataset with Sentinel-2 inputs and LULC label targets."""
    return ModelDataset(
        dataset=Dataset(path=ds_path),
        split_config=SplitConfig(
            groups=[GROUP],
            transforms=[
                OlmoEarthNormalize(band_names={"sentinel2_l2a": SENTINEL2_BANDS})
            ],
        ),
        inputs={
            "sentinel2_l2a": DataInput(
                data_type="raster",
                layers=["sentinel2"],
                bands=SENTINEL2_BANDS,
                passthrough=True,
                load_all_item_groups=True,
                load_all_layers=True,
            ),
            "label": DataInput(
                data_type="raster",
                layers=["lulc_label_raster"],
                bands=["lulc_label"],
                is_target=True,
                dtype=DType.INT32,
            ),
        },
        task=MultiTask(
            tasks={"segment": SegmentationTask(num_classes=3, zero_is_invalid=True)},
            input_mapping={"segment": {"label": "targets"}},
        ),
        workers=workers,
        fix_patch_pick=True,
    )


def process_batch(
    model: OlmoEarth,
    pixels: list[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Run a batch of single-pixel inputs through the model.

    Args:
        model: the OlmoEarth encoder.
        pixels: list of tensors each with shape (C*T, 1, 1).
        device: the torch device to use.

    Returns:
        numpy array of shape (batch_size, embedding_dim).
    """
    batch_inputs = [{"sentinel2_l2a": p.to(device)} for p in pixels]
    output = model(batch_inputs)
    return output[0][:, :, 0, 0].float().cpu().numpy()


def main() -> None:
    """Compute embeddings."""
    parser = argparse.ArgumentParser(
        description="Compute OlmoEarth nano embeddings for central pixels."
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the rslearn dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save embeddings.npy and metadata.csv",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of workers for dataset loading",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading OlmoEarth nano model...")
    model = OlmoEarth(model_id=ModelID.OLMOEARTH_V1_NANO, patch_size=1)
    model = model.to(device)
    model.eval()

    print("Loading dataset...")
    dataset = load_dataset(UPath(args.ds_path), args.workers)
    windows = dataset.get_dataset_examples()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    metadata_rows: list[dict[str, str]] = []
    batch_pixels: list[torch.Tensor] = []
    batch_window_indices: list[int] = []
    num_skipped = 0

    print(f"Processing {len(dataset)} windows...")
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset)), desc="Computing embeddings"):
            try:
                input_dict, _, _ = dataset[i]
            except Exception as e:
                print(f"Skipping window {i}: {e}")
                num_skipped += 1
                continue

            s2 = input_dict["sentinel2_l2a"]
            h, w = s2.shape[1], s2.shape[2]
            cx, cy = h // 2, w // 2
            pixel = s2[:, cx : cx + 1, cy : cy + 1]

            batch_pixels.append(pixel)
            batch_window_indices.append(i)

            if len(batch_pixels) == args.batch_size or i == len(dataset) - 1:
                batch_emb = process_batch(model, batch_pixels, device)
                for j, idx in enumerate(batch_window_indices):
                    window = windows[idx]
                    opts = window.options or {}
                    embeddings.append(batch_emb[j])
                    metadata_rows.append(
                        {
                            "window_name": window.name,
                            "lulc_label": opts.get("lulc_category", ""),
                            "crop_type": opts.get("crop_type", ""),
                            "split": opts.get("split", ""),
                        }
                    )
                batch_pixels = []
                batch_window_indices = []

    if num_skipped:
        print(f"Skipped {num_skipped} windows due to errors.")

    embeddings_arr = np.stack(embeddings)
    np.save(output_dir / "embeddings.npy", embeddings_arr)
    print(f"Saved embeddings with shape {embeddings_arr.shape}")

    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"Saved metadata to {csv_path}")

    print(f"Done. {len(embeddings)} embeddings saved to {output_dir}")


if __name__ == "__main__":
    main()
