"""Change detection from OlmoEarth embeddings.

Computes per-pixel cosine distance between two embedding COGs from different
time periods and generates a 1x3 figure: S2 RGB (before) | S2 RGB (after) |
change magnitude heatmap.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.analyze \
        --before-dir data/change_detection/2023 \
        --after-dir data/change_detection/2025 \
        --out figures/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.figure import Figure

from olmoearth_embeddings_tutorial.common.constants import (
    EMBEDDINGS_FILENAME,
    S2_RGB_FILENAME,
)
from olmoearth_embeddings_tutorial.common.embedding_utils import load_embeddings
from olmoearth_embeddings_tutorial.common.rgb_stretch import StretchMode, stretch_rgb


def _load_s2_rgb(
    path: Path,
    stretch: StretchMode = "percentile",
) -> np.ndarray | None:
    """Load S2 RGB and stretch to [0, 1]."""
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        rgb = ds.read([1, 2, 3]).astype(np.float32)
    rgb = np.moveaxis(rgb, 0, -1)
    return stretch_rgb(rgb, mode=stretch)


def _find_embeddings(directory: Path) -> Path:
    """Return the path to the embedding COG in *directory*."""
    path = directory / EMBEDDINGS_FILENAME
    if path.exists():
        return path
    raise FileNotFoundError(f"No {EMBEDDINGS_FILENAME} found in {directory}")


# ---------------------------------------------------------------------------
# Public API (used by both CLI and notebooks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangeResult:
    """Intermediate results from :func:`compute_change`."""

    change: np.ndarray
    valid: np.ndarray
    s2_before: np.ndarray | None
    s2_after: np.ndarray | None
    before_label: str
    after_label: str


def compute_change(
    before_dir: Path,
    after_dir: Path,
    before_label: str = "2023 (before)",
    after_label: str = "2025 (after)",
    stretch: StretchMode = "percentile",
) -> ChangeResult:
    """Compute per-pixel cosine distance between two embedding sets.

    Returns a :class:`ChangeResult` with the change magnitude array and
    all data needed for figure generation.
    """
    before_path = _find_embeddings(before_dir)
    after_path = _find_embeddings(after_dir)
    print(f"Before embeddings: {before_path}")
    print(f"After embeddings:  {after_path}")

    print("Loading before embeddings...")
    emb_b, valid_b, ds_b = load_embeddings(before_path)
    cb, hb, wb = emb_b.shape
    print(f"  {cb} bands, {hb}x{wb}, valid: {valid_b.sum():,}")

    print("Loading after embeddings...")
    emb_a, valid_a, ds_a = load_embeddings(after_path)
    ca, ha, wa = emb_a.shape
    print(f"  {ca} bands, {ha}x{wa}, valid: {valid_a.sum():,}")

    if (hb, wb) != (ha, wa):
        raise ValueError(
            f"Grid mismatch: before is {hb}x{wb}, after is {ha}x{wa}. "
            "Both predictions must use the same AOI."
        )

    valid = valid_b & valid_a

    print("Computing cosine distance...")
    eps = 1e-8
    norm_b = np.linalg.norm(emb_b, axis=0, keepdims=True).clip(eps)
    norm_a = np.linalg.norm(emb_a, axis=0, keepdims=True).clip(eps)
    cosine_sim = (emb_b / norm_b * emb_a / norm_a).sum(axis=0)
    change = 1.0 - cosine_sim
    change[~valid] = np.nan

    p50 = np.nanmedian(change[valid])
    p95 = np.nanpercentile(change[valid], 95)
    p99 = np.nanpercentile(change[valid], 99)
    print(f"  median={p50:.4f}, p95={p95:.4f}, p99={p99:.4f}")

    ds_b.close()
    ds_a.close()

    s2_before = _load_s2_rgb(before_dir / S2_RGB_FILENAME, stretch=stretch)
    s2_after = _load_s2_rgb(after_dir / S2_RGB_FILENAME, stretch=stretch)

    return ChangeResult(
        change=change,
        valid=valid,
        s2_before=s2_before,
        s2_after=s2_after,
        before_label=before_label,
        after_label=after_label,
    )


def make_change_figure(result: ChangeResult) -> Figure:
    """Create a 1x3 change detection figure.

    Returns the matplotlib Figure (caller decides whether to save or show).
    """
    ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5))
    fig.subplots_adjust(left=0.02, right=0.92, top=0.88, bottom=0.08, wspace=0.06)

    if result.s2_before is not None:
        axes[0].imshow(result.s2_before, interpolation="nearest")
    else:
        axes[0].set_facecolor("#1f1f1f")
    axes[0].set_title(result.before_label, fontsize=13, fontweight="bold", pad=6)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    if result.s2_after is not None:
        axes[1].imshow(result.s2_after, interpolation="nearest")
    else:
        axes[1].set_facecolor("#1f1f1f")
    axes[1].set_title(result.after_label, fontsize=13, fontweight="bold", pad=6)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    change_display = np.where(result.valid, result.change, np.nan)
    vmin = np.nanpercentile(change_display, 1)
    vmax = np.nanpercentile(change_display, 99)
    im = axes[2].imshow(
        change_display,
        cmap="inferno",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_title("Change magnitude", fontsize=13, fontweight="bold", pad=6)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.06, shrink=0.85)
    cbar.set_label("Cosine distance", fontsize=11)

    fig.suptitle(
        "Wildfire change detection from OlmoEarth embeddings",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-dir",
        type=Path,
        required=True,
        help="Directory with before-period data (embeddings, s2_rgb.tif).",
    )
    parser.add_argument(
        "--after-dir",
        type=Path,
        required=True,
        help="Directory with after-period data (embeddings, s2_rgb.tif).",
    )
    parser.add_argument(
        "--before-label",
        default="2023 (before)",
        help="Label for the before panel.",
    )
    parser.add_argument(
        "--after-label",
        default="2025 (after)",
        help="Label for the after panel.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="Output directory for the figure.",
    )
    parser.add_argument(
        "--stretch",
        choices=["percentile", "fixed"],
        default="percentile",
        help="RGB stretch mode: 'percentile' (2/98) or 'fixed' ([0, 0.25] reflectance).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return parser.parse_args()


def main() -> None:
    """Compute change detection and save the figure."""
    args = parse_args()
    result = compute_change(
        args.before_dir,
        args.after_dir,
        args.before_label,
        args.after_label,
        args.stretch,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    fig = make_change_figure(result)
    fig_path = args.out / "change_detection.png"
    fig.savefig(
        fig_path,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=True,
    )
    plt.close(fig)
    print(f"Wrote {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
