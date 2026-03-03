#!/usr/bin/env python3
"""Detect grid artifacts in discrete classification GeoTIFFs.

Computes a pixel-transition difference map, then analyzes row/column average
transition rates for periodic spikes that indicate tiling seams.

Usage:
    python detect_grid_artifacts.py input.tif [--max-index 5000] [--output report.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import stats


def compute_diff_map(data: np.ndarray, nodata_val: float | None) -> np.ndarray:
    """Compute a binary difference map (1 where adjacent pixels differ).

    Transitions involving nodata pixels are excluded (set to 0).
    """
    nodata_mask = np.zeros_like(data, dtype=bool)
    if nodata_val is not None:
        nodata_mask = data == nodata_val

    diff = np.zeros(data.shape, dtype=np.uint8)

    # Horizontal: data[i,j] != data[i,j+1], excluding nodata on either side
    h_diff = (data[:, :-1] != data[:, 1:]) & ~nodata_mask[:, :-1] & ~nodata_mask[:, 1:]
    diff[:, :-1] |= h_diff

    # Vertical: data[i,j] != data[i+1,j], excluding nodata on either side
    v_diff = (data[:-1, :] != data[1:, :]) & ~nodata_mask[:-1, :] & ~nodata_mask[1:, :]
    diff[:-1, :] |= v_diff

    return diff


def compute_averages(
    diff: np.ndarray, nodata_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute row and column averages of the diff map, ignoring nodata pixels.

    Fully vectorized using masked arrays.
    """
    masked = np.ma.array(diff.astype(np.float32), mask=nodata_mask)
    col_avgs = masked.mean(axis=0).filled(0.0)
    row_avgs = masked.mean(axis=1).filled(0.0)
    return row_avgs, col_avgs


def find_spikes_local(avgs: np.ndarray, neighbor_factor: float = 1.5) -> np.ndarray:
    """Find spikes using local maxima: points greater than `neighbor_factor`×
    both neighbors. This is the approach from the original notebook."""
    if len(avgs) < 3:
        return np.array([], dtype=int)
    left = avgs[:-2]
    center = avgs[1:-1]
    right = avgs[2:]
    mask = (center > neighbor_factor * left) & (center > neighbor_factor * right)
    return np.where(mask)[0] + 1


def find_spikes(avgs: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """Find spike indices using median absolute deviation (MAD) z-scores.

    This is more robust than the simple 1.5x neighbor threshold used in the
    notebook, because it doesn't depend on local context and is resistant to
    outliers in the baseline.
    """
    median = np.median(avgs)
    mad = np.median(np.abs(avgs - median))
    if mad == 0:
        # Fall back to std-based z-score if MAD is zero
        std = np.std(avgs)
        if std == 0:
            return np.array([], dtype=int)
        z_scores = (avgs - np.mean(avgs)) / std
    else:
        # Modified z-score (0.6745 is the 0.75th quantile of the normal dist)
        z_scores = 0.6745 * (avgs - median) / mad

    # Only consider positive spikes (above the baseline)
    spike_mask = z_scores > z_thresh

    # Additionally require local maximum (greater than both neighbors)
    local_max = np.zeros_like(spike_mask)
    local_max[1:-1] = (avgs[1:-1] > avgs[:-2]) & (avgs[1:-1] > avgs[2:])
    spike_mask &= local_max

    return np.where(spike_mask)[0]


def analyze_spacing(maxima: np.ndarray, label: str) -> dict:
    """Analyze regularity of inter-maxima spacing."""
    if len(maxima) < 2:
        return {"label": label, "n_maxima": len(maxima), "spacings": np.array([])}

    spacings = np.diff(maxima)
    unique, counts = np.unique(spacings, return_counts=True)

    # Sort by frequency descending
    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]

    # Regularity score: fraction of spacings accounted for by the top-1 value
    top_fraction = counts[0] / len(spacings) if len(spacings) > 0 else 0.0

    # Chi-squared test: are the spacings uniformly distributed across the
    # observed unique values? Under H0 (no periodicity), each unique spacing
    # should appear with roughly equal frequency. A tiny p-value means the
    # distribution is significantly non-uniform → some spacing dominates.
    if len(unique) >= 2:
        expected = np.full_like(counts, fill_value=len(spacings) / len(unique), dtype=float)
        chi2, p_value = stats.chisquare(counts.astype(float), f_exp=expected)
    else:
        chi2, p_value = 0.0, 1.0

    return {
        "label": label,
        "n_maxima": len(maxima),
        "spacings": spacings,
        "top_spacings": list(zip(unique[:10].tolist(), counts[:10].tolist())),
        "regularity_score": top_fraction,
        "chi2": chi2,
        "p_value": p_value,
    }


def generate_report(
    row_avgs: np.ndarray,
    col_avgs: np.ndarray,
    row_spikes: np.ndarray,
    col_spikes: np.ndarray,
    row_analysis: dict,
    col_analysis: dict,
    max_index: int,
    output_path: str,
):
    """Generate a multi-panel report figure."""
    fig = plt.figure(figsize=(16, 10))
    # Top row: single wide axes spanning both columns
    ax_top = fig.add_subplot(2, 1, 1)
    # Bottom row: two side-by-side axes
    ax_bl = fig.add_subplot(2, 2, 3)
    ax_br = fig.add_subplot(2, 2, 4)

    row_n = min(len(row_avgs), max_index)
    col_n = min(len(col_avgs), max_index)

    # --- Top: averages with spikes marked ---
    ax = ax_top
    ax.plot(np.arange(col_n), col_avgs[:col_n], linewidth=0.5, alpha=0.6, color="C0")
    col_sp_vis = col_spikes[col_spikes < col_n]
    ax.scatter(col_sp_vis, col_avgs[col_sp_vis], color="C0", s=12, zorder=5, label=f"Col spikes ({len(col_spikes)} total)")

    ax.plot(np.arange(row_n), row_avgs[:row_n], linewidth=0.5, alpha=0.6, color="C1")
    row_sp_vis = row_spikes[row_spikes < row_n]
    ax.scatter(row_sp_vis, row_avgs[row_sp_vis], color="C1", s=12, zorder=5, label=f"Row spikes ({len(row_spikes)} total)")

    ax.set_ylabel("Mean transition rate")
    ax.set_xlabel("Index")
    ax.set_title(f"Transition averages with detected spikes (first {max_index} indices)")
    ax.legend()
    ax.set_ylim(bottom=0)

    # --- Bottom left: column spacing top-10 bar chart ---
    ax = ax_bl
    if "top_spacings" in col_analysis and len(col_analysis["top_spacings"]) > 0:
        top = col_analysis["top_spacings"][:10]
        labels = [str(s) for s, _ in top]
        counts = [c for _, c in top]
        ax.bar(range(len(top)), counts, tick_label=labels, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Col spike spacings — top {len(top)} most frequent\nRegularity: {col_analysis['regularity_score']:.2f}  χ² p={col_analysis['p_value']:.2e}")
    else:
        ax.set_title("Col spike spacings — insufficient data")
    ax.set_xlabel("Spacing (pixels)")
    ax.set_ylabel("Count")

    # --- Bottom right: row spacing top-10 bar chart ---
    ax = ax_br
    if "top_spacings" in row_analysis and len(row_analysis["top_spacings"]) > 0:
        top = row_analysis["top_spacings"][:10]
        labels = [str(s) for s, _ in top]
        counts = [c for _, c in top]
        ax.bar(range(len(top)), counts, tick_label=labels, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Row spike spacings — top {len(top)} most frequent\nRegularity: {row_analysis['regularity_score']:.2f}  χ² p={row_analysis['p_value']:.2e}")
    else:
        ax.set_title("Row spike spacings — insufficient data")
    ax.set_xlabel("Spacing (pixels)")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect grid artifacts in discrete classification GeoTIFFs."
    )
    parser.add_argument("input", help="Path to input GeoTIFF")
    parser.add_argument(
        "--max-index",
        type=int,
        default=5000,
        help="Max row/col index to show in plots (default: 5000)",
    )
    parser.add_argument(
        "--spike-method",
        choices=["mad", "local"],
        default="mad",
        help="Spike detection method: 'mad' (MAD z-score, default) or 'local' (neighbor_factor x both neighbors)",
    )
    parser.add_argument(
        "--neighbor-factor",
        type=float,
        default=1.5,
        help="Factor for local spike detection: spike if value > factor x both neighbors (default: 1.5)",
    )
    parser.add_argument(
        "--z-thresh",
        type=float,
        default=3.0,
        help="MAD z-score threshold for spike detection (default: 3.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: <input>_grid_artifact_report.png)",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=1,
        help="Band number to analyze (default: 1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(input_path.with_suffix("")) + "_grid_artifact_report.png"

    # 1. Read data
    print(f"Reading {input_path} (band {args.band})...")
    with rasterio.open(input_path) as src:
        data = src.read(args.band)
        nodata_val = src.nodata

    print(f"  Shape: {data.shape}, dtype: {data.dtype}, nodata: {nodata_val}")

    # 2. Compute difference map
    print("Computing difference map...")
    diff = compute_diff_map(data, nodata_val)

    # 3. Compute row/col averages (vectorized)
    print("Computing row/column averages...")
    nodata_mask = np.zeros_like(data, dtype=bool)
    if nodata_val is not None:
        nodata_mask = data == nodata_val
    row_avgs, col_avgs = compute_averages(diff, nodata_mask)

    # 4. Find spikes
    method = args.spike_method
    if method == "mad":
        print(f"Finding spikes (method=mad, z-thresh={args.z_thresh})...")
        row_spikes = find_spikes(row_avgs, z_thresh=args.z_thresh)
        col_spikes = find_spikes(col_avgs, z_thresh=args.z_thresh)
    else:
        print(f"Finding spikes (method=local, neighbor-factor={args.neighbor_factor})...")
        row_spikes = find_spikes_local(row_avgs, neighbor_factor=args.neighbor_factor)
        col_spikes = find_spikes_local(col_avgs, neighbor_factor=args.neighbor_factor)
    print(f"  Row spikes: {len(row_spikes)}, Col spikes: {len(col_spikes)}")

    # 5. Analyze spacing regularity
    row_analysis = analyze_spacing(row_spikes, "Row")
    col_analysis = analyze_spacing(col_spikes, "Column")

    # 6. Print summary
    print("\n--- Summary ---")
    for analysis in [row_analysis, col_analysis]:
        label = analysis["label"]
        print(f"{label}: {analysis['n_maxima']} spikes detected")
        if "top_spacings" in analysis:
            print(f"  Top spacings (value, count): {analysis['top_spacings']}")
            score = analysis["regularity_score"]
            p = analysis["p_value"]
            print(f"  Regularity score: {score:.2f}  |  Chi-squared p-value: {p:.2e}")
            if p < 0.01 and score > 0.3:
                print(f"  ⚠️  HIGH — likely grid artifacts (dominant spacing accounts for {score:.0%} of gaps, p={p:.2e})")
            elif p < 0.05:
                print(f"  ⚡ MODERATE — possible grid artifacts (p={p:.2e})")
            else:
                print(f"  ✓ LOW — no significant periodicity (p={p:.2e})")

    # 7. Generate report plot
    print(f"\nGenerating report...")
    generate_report(
        row_avgs, col_avgs,
        row_spikes, col_spikes,
        row_analysis, col_analysis,
        max_index=args.max_index,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
