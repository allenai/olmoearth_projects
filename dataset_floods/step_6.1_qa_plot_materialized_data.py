"""
Quality assurance visualization of materialized training data.
Plots label masks, Sentinel-1 VV, and Sentinel-2 RGB for selected chips.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap


# Explicit list of chips you want to visualize
CHIPS = [
    "Sri-Lanka_551926_chip0",
    "Sri-Lanka_120804_chip0",
    "Cambodia_221789_chip0",
    "Pakistan_548910_chip0",
    "Somalia_230192_chip0",
    "USA_486103_chip0",
    "Spain_2594119_chip0",
]


# Base directory for default windows
BASE_DIR = Path(
    "dataset_floods/windows/default"
)


# Utility — normalize band to [0,1]
def normalize_band(band: np.ndarray) -> np.ndarray:
    lo = np.percentile(band, 2)
    hi = np.percentile(band, 98)
    return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)


# Process a single chip
def plot_chip(chip_id: str) -> None:
    chip_dir = BASE_DIR / chip_id
    layers_dir = chip_dir / "layers"

    # File paths
    label_tif = layers_dir / "label/B1/geotiff.tif"
    s1_tif = layers_dir / "sentinel1/vv_vh/geotiff.tif"
    s2_tif = layers_dir / (
        "sentinel2_l2a/"
        "B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif"
    )
    items_json = chip_dir / "items.json"

    # Validate files exist
    for p in [label_tif, s1_tif, s2_tif, items_json]:
        if not p.exists():
            print(f"[SKIP] Missing {p} for chip {chip_id}")
            return

    # Safe timestamp extraction from items.json
    with items_json.open() as f:
        items = json.load(f)

    s1_date = "Unknown"
    s2_date = "Unknown"

    for item in items:
        layer_name = item.get("layer_name", "")
        groups = item.get("serialized_item_groups")

        if not groups or not groups[0]:
            print(f"[WARN] Empty item groups for {chip_id} ({layer_name})")
            continue

        geom = groups[0][0].get("geometry", {})
        time_range = geom.get("time_range")

        if not time_range or not isinstance(time_range, list):
            print(f"[WARN] Missing time_range for {chip_id} ({layer_name})")
            continue

        date = time_range[0][:10]

        if layer_name == "sentinel1":
            s1_date = date
        elif layer_name == "sentinel2_l2a":
            s2_date = date

    # Load imagery
    with rasterio.open(label_tif) as src:
        label = src.read(1)

    with rasterio.open(s1_tif) as src:
        vv = src.read(1)

    with rasterio.open(s2_tif) as src:
        s2 = src.read()  # shape: (13 bands, H, W)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Label
    label_mask = np.ma.masked_where(label == 255, label)
    cmap = ListedColormap(["lightgray", "blue"])
    axes[0].imshowlabel = axes[0].imshow(
        label_mask, cmap=cmap, vmin=0, vmax=1
    )
    axes[0].set_title("Label", fontsize=11)
    axes[0].axis("off")

    # Sentinel-1 VV
    vv_lo = np.percentile(vv, 2)
    vv_hi = np.percentile(vv, 98)
    axes[1].imshow(vv, cmap="gray", vmin=vv_lo, vmax=vv_hi)
    axes[1].set_title(f"Sentinel-1 VV\n{s1_date}", fontsize=11)
    axes[1].axis("off")

    # Sentinel-2 RGB (B04, B03, B02)
    rgb = np.dstack([
        normalize_band(s2[3]),  # Red
        normalize_band(s2[2]),  # Green
        normalize_band(s2[1])   # Blue
    ])
    axes[2].imshow(rgb)
    axes[2].set_title(f"Sentinel-2 RGB\n{s2_date}", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()

    # Save output
    outfile = f"chip_visual_{chip_id}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved {outfile}")


# Main loop
print(f"Processing {len(CHIPS)} chips...\n")

for chip in CHIPS:
    plot_chip(chip)

print("\nAll visualizations completed.\n")