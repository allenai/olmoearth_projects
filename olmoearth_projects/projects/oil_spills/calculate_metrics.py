#!/usr/bin/env python3
"""Calculate confusion matrix and metrics for segmentation predictions.

This script compares ground truth labels with model predictions and calculates
pixel-wise metrics. It works for any binary or multi-class segmentation task.

TODO: For point label datasets, we need to handle random cropping:
- Point labels use random crops (e.g., 16x16 from 31x31) during training/val
- Predictions should use the same cropping to match training distribution
- Need to save/load crop metadata (bounds/offsets) to align predictions with labels
- Currently only works for polygon labels where predictions match label dimensions
"""

import json
import numpy as np
import rasterio
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple
import shutil

# ============================================================================
# CONFIGURATION - Modify these for your task
# ============================================================================

# Class configuration
POSITIVE_CLASS_VALUE = 1  # Value representing the positive class (e.g., oil slick)
NEGATIVE_CLASS_VALUE = 0  # Value representing the negative class (background)
NODATA_VALUE = 255  # Value representing nodata/invalid pixels to exclude

CLASS_NAMES = {
    NEGATIVE_CLASS_VALUE: "No Oil",
    POSITIVE_CLASS_VALUE: "Oil Slick"
}

# Paths
LABEL_LAYER_NAME = "label"
PREDICTION_LAYER_NAME = "output"

# Example saving
N_EXAMPLES_PER_CATEGORY = 5  # Number of example windows to save for each category


# ============================================================================
# Helper Functions
# ============================================================================

def find_validation_windows(base_path: Path) -> List[Dict]:
    """Find all validation windows with both labels and predictions.

    Returns:
        List of dicts with 'name', 'label_file', 'pred_file' keys
    """
    windows = []

    # Find all windows with predictions
    output_dirs = list(base_path.glob(f"*/layers/{PREDICTION_LAYER_NAME}"))

    for output_dir in output_dirs:
        window_dir = output_dir.parent.parent

        # Find label and prediction files
        label_files = list((window_dir / "layers" / LABEL_LAYER_NAME).glob("*/geotiff.tif"))
        pred_files = list((window_dir / "layers" / PREDICTION_LAYER_NAME).glob("*/geotiff.tif"))

        if label_files and pred_files:
            windows.append({
                'name': window_dir.name,
                'label_file': label_files[0],
                'pred_file': pred_files[0]
            })

    return windows


def load_and_align_data(label_file: Path, pred_file: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load label and prediction, check alignment, and return valid pixels.

    Returns:
        (label_valid, pred_valid, metadata) where metadata contains dimensions/bounds
    """
    # Load label
    with rasterio.open(label_file) as src:
        label = src.read(1)
        label_bounds = src.bounds
        label_shape = label.shape

    # Load prediction
    with rasterio.open(pred_file) as src:
        pred = src.read(1)
        pred_bounds = src.bounds
        pred_shape = pred.shape

    # Check alignment
    if label_shape != pred_shape:
        raise ValueError(
            f"Shape mismatch: label {label_shape} != prediction {pred_shape}\n"
            "For point labels, see TODO at top of script about handling random crops."
        )

    if label_bounds != pred_bounds:
        raise ValueError(
            f"Bounds mismatch: label {label_bounds} != prediction {pred_bounds}"
        )

    # Filter out nodata pixels
    valid_mask = label != NODATA_VALUE
    label_valid = label[valid_mask]
    pred_valid = pred[valid_mask]

    metadata = {
        'shape': label_shape,
        'bounds': label_bounds,
        'n_valid_pixels': np.sum(valid_mask),
        'n_nodata_pixels': np.sum(~valid_mask)
    }

    return label_valid, pred_valid, metadata


def categorize_window(label_valid: np.ndarray, pred_valid: np.ndarray) -> str:
    """Categorize window based on presence of positive class in label/prediction.

    Returns:
        'TP': Has positive in both label and prediction
        'FN': Has positive in label, none in prediction
        'FP': No positive in label, has positive in prediction
        'TN': No positive in either
    """
    has_positive_label = np.any(label_valid == POSITIVE_CLASS_VALUE)
    has_positive_pred = np.any(pred_valid == POSITIVE_CLASS_VALUE)

    if has_positive_label and has_positive_pred:
        return 'TP'
    elif has_positive_label and not has_positive_pred:
        return 'FN'
    elif not has_positive_label and has_positive_pred:
        return 'FP'
    else:
        return 'TN'


def save_example_windows(windows: List[Dict], categories: Dict[str, List[str]],
                        output_dir: Path, n_per_category: int = 5):
    """Save example windows for each category to output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for category, window_names in categories.items():
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)

        # Save up to n examples
        for i, window_name in enumerate(window_names[:n_per_category]):
            # Find the window
            window = next((w for w in windows if w['name'] == window_name), None)
            if window is None:
                continue

            # Copy label and prediction files
            example_dir = category_dir / f"{i+1:02d}_{window_name}"
            example_dir.mkdir(exist_ok=True)

            shutil.copy2(window['label_file'], example_dir / "label.tif")
            shutil.copy2(window['pred_file'], example_dir / "prediction.tif")

        print(f"  Saved {min(len(window_names), n_per_category)} examples for {category} category")


# ============================================================================
# Main Calculation
# ============================================================================

def calculate_metrics(base_path: Path, save_examples: bool = True,
                     examples_dir: Path = None) -> Dict:
    """Calculate confusion matrix and metrics across all validation windows."""

    windows = find_validation_windows(base_path)
    print(f"Found {len(windows)} validation windows with predictions\n")

    if len(windows) == 0:
        raise ValueError("No windows found with both labels and predictions")

    # Accumulate all predictions and labels
    all_labels = []
    all_preds = []

    # Track window categories for examples
    window_categories = {'TP': [], 'FN': [], 'FP': [], 'TN': []}

    # Statistics
    total_valid_pixels = 0
    total_nodata_pixels = 0

    for i, window in enumerate(windows):
        try:
            label_valid, pred_valid, metadata = load_and_align_data(
                window['label_file'],
                window['pred_file']
            )

            # Categorize window
            category = categorize_window(label_valid, pred_valid)
            window_categories[category].append(window['name'])

            # Accumulate
            all_labels.extend(label_valid.flatten())
            all_preds.extend(pred_valid.flatten())

            total_valid_pixels += metadata['n_valid_pixels']
            total_nodata_pixels += metadata['n_nodata_pixels']

            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(windows)} windows...")

        except Exception as e:
            print(f"Error processing {window['name']}: {e}")
            continue

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Print window-level statistics
    print(f"\n{'='*70}")
    print("WINDOW-LEVEL STATISTICS")
    print(f"{'='*70}")
    print(f"Total windows processed: {len(windows)}")
    print(f"  TP windows (has positive in both):     {len(window_categories['TP']):3d}")
    print(f"  FN windows (positive in label only):   {len(window_categories['FN']):3d}")
    print(f"  FP windows (positive in pred only):    {len(window_categories['FP']):3d}")
    print(f"  TN windows (no positive in either):    {len(window_categories['TN']):3d}")

    # Print pixel statistics
    print(f"\n{'='*70}")
    print("PIXEL STATISTICS")
    print(f"{'='*70}")
    print(f"Total pixels analyzed: {len(all_labels):,} (excludes nodata)")
    print(f"Nodata pixels excluded: {total_nodata_pixels:,}")
    print(f"Positive class pixels in labels: {np.sum(all_labels == POSITIVE_CLASS_VALUE):,}")
    print(f"Positive class pixels in predictions: {np.sum(all_preds == POSITIVE_CLASS_VALUE):,}")

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[NEGATIVE_CLASS_VALUE, POSITIVE_CLASS_VALUE])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*70}")
    print("CONFUSION MATRIX (Pixel-Level)")
    print(f"{'='*70}")
    print(f"                    Predicted: {CLASS_NAMES[NEGATIVE_CLASS_VALUE]:15s} | Predicted: {CLASS_NAMES[POSITIVE_CLASS_VALUE]:15s}")
    print(f"Actual: {CLASS_NAMES[NEGATIVE_CLASS_VALUE]:15s} {tn:>15,}  | {fp:>15,}")
    print(f"Actual: {CLASS_NAMES[POSITIVE_CLASS_VALUE]:15s} {fn:>15,}  | {tp:>15,}")

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    print(f"\n{'='*70}")
    print(f"METRICS FOR {CLASS_NAMES[POSITIVE_CLASS_VALUE].upper()}")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print(f"True Positives (TP):  {tp:>12,} pixels correctly identified as {CLASS_NAMES[POSITIVE_CLASS_VALUE].lower()}")
    print(f"True Negatives (TN):  {tn:>12,} pixels correctly identified as {CLASS_NAMES[NEGATIVE_CLASS_VALUE].lower()}")
    print(f"False Positives (FP): {fp:>12,} pixels incorrectly flagged as {CLASS_NAMES[POSITIVE_CLASS_VALUE].lower()}")
    print(f"False Negatives (FN): {fn:>12,} pixels of {CLASS_NAMES[POSITIVE_CLASS_VALUE].lower()} that were missed")

    # Warnings
    print()
    if recall < 0.5:
        print(f"⚠️  LOW RECALL ({recall:.1%}) - Model is missing most positive class pixels")
    if precision < 0.5:
        print(f"⚠️  LOW PRECISION ({precision:.1%}) - Model has many false alarms")
    if f1 < 0.5:
        print(f"⚠️  LOW F1-SCORE ({f1:.3f}) - Overall poor performance")

    # Detailed classification report
    print(f"\n{'='*70}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'='*70}")
    print(classification_report(
        all_labels, all_preds,
        target_names=[CLASS_NAMES[NEGATIVE_CLASS_VALUE], CLASS_NAMES[POSITIVE_CLASS_VALUE]],
        labels=[NEGATIVE_CLASS_VALUE, POSITIVE_CLASS_VALUE],
        digits=4
    ))

    # Save examples
    if save_examples:
        if examples_dir is None:
            examples_dir = base_path.parent / "metric_examples"

        print(f"\n{'='*70}")
        print(f"SAVING EXAMPLE WINDOWS to {examples_dir}")
        print(f"{'='*70}")
        save_example_windows(windows, window_categories, examples_dir, N_EXAMPLES_PER_CATEGORY)

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'n_windows': len(windows),
        'n_pixels': len(all_labels),
        'window_categories': {k: len(v) for k, v in window_categories.items()}
    }


if __name__ == "__main__":
    import sys

    # Default path
    base_path = Path("~/olmoearth_projects/oerun_dataset/oil_spills/dataset/windows/spatial_split_10km").expanduser()

    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])

    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        sys.exit(1)

    print(f"Calculating metrics for validation set...")
    print(f"Base path: {base_path}")
    print(f"Label layer: {LABEL_LAYER_NAME}")
    print(f"Prediction layer: {PREDICTION_LAYER_NAME}")
    print(f"Positive class value: {POSITIVE_CLASS_VALUE}")
    print(f"Nodata value: {NODATA_VALUE}\n")

    metrics = calculate_metrics(base_path)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
