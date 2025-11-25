import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# -------------------------------------------------------------
# Chips to visualize
# -------------------------------------------------------------
CHIPS = [
    "Cambodia_333434_chip0",
    "Ghana_97059_chip0",
    "India_79637_chip0",
    "India_1018327_chip0",
]


# -------------------------------------------------------------
# Base directories
# -------------------------------------------------------------
PREDICT_BASE = Path(
    "dataset_floods/windows/predict"
)

GT_BASE = Path(
    "sen1floods11_data/data/LabelHand"
)


# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------
def normalize_band(band: np.ndarray) -> np.ndarray:
    lo = np.percentile(band, 2)
    hi = np.percentile(band, 98)
    return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)


def convert_mask(x: np.ndarray) -> np.ndarray:
    # -1 -> 0 (gray), 0 -> 1 (blue), 1 -> 2 (red)
    return np.where(x == -1, 0, np.where(x == 0, 1, 2))


MASK_COLORS = ["gray", "blue", "red"]
MASK_CMAP = ListedColormap(MASK_COLORS)


# -------------------------------------------------------------
# Process a single chip
# -------------------------------------------------------------
def process_chip(chip_id: str) -> None:
    chip_dir = PREDICT_BASE / chip_id / "layers"

    prediction_tif = chip_dir / "output/output/geotiff.tif"
    s1_tif = chip_dir / "sentinel1/vv_vh/geotiff.tif"
    s2_tif = chip_dir / (
        "sentinel2_l2a/"
        "B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif"
    )

    label_tif = GT_BASE / f"{chip_id}_LabelHand.tif"

    # Validate files exist
    for p in [prediction_tif, s1_tif, s2_tif, label_tif]:
        if not p.exists():
            print(f"[SKIP] Missing file: {p}")
            return

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    with rasterio.open(label_tif) as src:
        label = src.read(1)

    with rasterio.open(prediction_tif) as src:
        pred = src.read(1)

    with rasterio.open(s1_tif) as src:
        vv = src.read(1)

    with rasterio.open(s2_tif) as src:
        s2 = src.read()
        rgb = np.dstack([
            normalize_band(s2[3]),  # B04
            normalize_band(s2[2]),  # B03
            normalize_band(s2[1]),  # B02
        ])

    # ---------------------------------------------------------
    # Flood percentages
    # ---------------------------------------------------------
    valid_gt = label != -1
    valid_pred = pred != -1

    flood_pct_gt = (np.sum(label == 1) / np.sum(valid_gt)) * 100
    flood_pct_pr = (np.sum(pred == 1) / np.sum(valid_pred)) * 100

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Sentinel-2 RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Sentinel-2 RGB", fontsize=13)
    axes[0, 0].axis("off")

    # Sentinel-1 VV
    vv_lo = np.percentile(vv, 2)
    vv_hi = np.percentile(vv, 98)
    axes[0, 1].imshow(vv, cmap="gray", vmin=vv_lo, vmax=vv_hi)
    axes[0, 1].set_title("Sentinel-1 VV", fontsize=13)
    axes[0, 1].axis("off")

    # Ground truth mask
    axes[1, 0].imshow(convert_mask(label), cmap=MASK_CMAP, vmin=0, vmax=2)
    axes[1, 0].set_title(
        f"Ground Truth\nFlood: {flood_pct_gt:.1f}%",
        fontsize=13,
    )
    axes[1, 0].axis("off")

    # Prediction mask
    axes[1, 1].imshow(convert_mask(pred), cmap=MASK_CMAP, vmin=0, vmax=2)
    axes[1, 1].set_title(
        f"Prediction\nFlood: {flood_pct_pr:.1f}%",
        fontsize=13,
    )
    axes[1, 1].axis("off")

    # Legend
    legend_items = [
        Patch(facecolor="blue", label="No Flood (0)"),
        Patch(facecolor="red", label="Flood (1)"),
        Patch(facecolor="gray", label="NoData (-1)"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        fontsize=12,
    )

    plt.suptitle(f"{chip_id}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, top=0.94)

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------
    outfile = f"prediction_visual_{chip_id}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {outfile}")


# -------------------------------------------------------------
# Main loop
# -------------------------------------------------------------
print(f"Processing {len(CHIPS)} chips...\n")

for chip in CHIPS:
    process_chip(chip)

print("\nAll prediction visualizations complete.\n")
