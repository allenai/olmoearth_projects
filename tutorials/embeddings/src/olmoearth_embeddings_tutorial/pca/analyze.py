"""PCA false-color visualization of OlmoEarth embeddings.

Maps the first three principal components of each pixel's embedding vector to
R/G/B channels.  Pixels with similar embeddings receive similar colors, giving
a spatial "similarity image" that is directly comparable to Sentinel-2 RGB.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.pca.analyze \
        --embed data/flevoland/embeddings.tif \
        --rgb data/flevoland/s2_rgb.tif \
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from olmoearth_embeddings_tutorial.common.embedding_utils import load_embeddings
from olmoearth_embeddings_tutorial.common.rgb_stretch import StretchMode, stretch_rgb


def _load_s2_rgb(
    path: Path,
    target_shape: tuple[int, int],
    stretch: StretchMode = "percentile",
) -> np.ndarray | None:
    """Load S2 RGB and stretch to [0, 1] for display."""
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        rgb = ds.read([1, 2, 3]).astype(np.float32)
    h, w = target_shape
    if rgb.shape[1] != h or rgb.shape[2] != w:
        print(f"S2 shape {rgb.shape[1:]} != embed {h}x{w}; skipping S2 panel.")
        return None
    rgb = np.moveaxis(rgb, 0, -1)
    return stretch_rgb(rgb, mode=stretch)


# ---------------------------------------------------------------------------
# Public API (used by both CLI and notebooks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PCAResult:
    """Intermediate results from :func:`compute_pca_rgb`."""

    pca_rgb: np.ndarray
    valid: np.ndarray
    variance_pct: np.ndarray
    height: int
    width: int


def compute_pca_rgb(
    embed_path: Path,
    seed: int = 42,
) -> PCAResult:
    """Compute PCA false-color RGB from embeddings.

    Returns a :class:`PCAResult` with the false-color image and variance info.
    """
    print("Loading embeddings...")
    embed, valid, dataset = load_embeddings(embed_path)
    c, h, w = embed.shape
    flat = embed.reshape(c, -1).T
    valid_idx = np.flatnonzero(valid.ravel())
    print(f"  {c} bands, {h}x{w}, valid: {valid_idx.size:,}")
    dataset.close()

    scaler = StandardScaler()
    X_valid = scaler.fit_transform(flat[valid_idx])

    print("Running PCA (3 components)...")
    pca = PCA(n_components=3, random_state=seed)
    pc3 = pca.fit_transform(X_valid)
    var = pca.explained_variance_ratio_ * 100
    print(
        f"  PC1={var[0]:.1f}%  PC2={var[1]:.1f}%  "
        f"PC3={var[2]:.1f}%  total={var.sum():.1f}%"
    )

    pca_flat = np.full((h * w, 3), np.nan, dtype=np.float32)
    pca_flat[valid_idx] = pc3
    pca_rgb = pca_flat.reshape(h, w, 3).copy()

    for ch in range(3):
        vals = pca_rgb[:, :, ch][valid]
        lo, hi = np.percentile(vals, [2, 98])
        pca_rgb[:, :, ch] = (pca_rgb[:, :, ch] - lo) / max(float(hi - lo), 1e-6)
    pca_rgb = np.clip(pca_rgb, 0, 1)
    pca_rgb[~valid] = 1.0

    return PCAResult(
        pca_rgb=pca_rgb,
        valid=valid,
        variance_pct=var,
        height=h,
        width=w,
    )


def make_pca_figure(
    result: PCAResult,
    s2_rgb: np.ndarray | None = None,
) -> Figure:
    """Create a side-by-side S2 RGB / PCA false-color figure.

    If *s2_rgb* is ``None``, only the PCA panel is shown.
    Returns the matplotlib Figure (caller decides whether to save or show).
    """
    var = result.variance_pct
    has_s2 = s2_rgb is not None
    ncols = 2 if has_s2 else 1
    # Size each panel to the raster's aspect ratio, plus room for the titles
    # above and the variance note below.
    panel_h = 5.0
    panel_w = panel_h * result.width / result.height
    fig_w = panel_w * ncols * 1.06
    fig_h = panel_h + 1.1
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h))
    if ncols == 1:
        axes = [axes]
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=1 - 0.8 / fig_h,
        bottom=0.35 / fig_h,
        wspace=0.06,
    )

    col = 0
    if has_s2:
        axes[col].imshow(s2_rgb, interpolation="nearest")
        axes[col].set_title("Sentinel-2 RGB", fontsize=13, fontweight="bold", pad=6)
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        col += 1

    axes[col].imshow(result.pca_rgb, interpolation="nearest")
    axes[col].set_title(
        "Embedding PCA (PC1/PC2/PC3 \u2192 R/G/B)",
        fontsize=13,
        fontweight="bold",
        pad=6,
    )
    axes[col].set_xticks([])
    axes[col].set_yticks([])

    fig.suptitle(
        "OlmoEarth embedding structure", fontsize=15, fontweight="bold", y=0.97
    )
    fig.text(
        0.5,
        0.01,
        (
            f"PC1={var[0]:.1f}%   PC2={var[1]:.1f}%   PC3={var[2]:.1f}%   "
            f"(total {var.sum():.1f}% of variance)"
        ),
        ha="center",
        fontsize=11,
        color="0.35",
    )

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embed",
        type=Path,
        required=True,
        help="Path to the embedding COG.",
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        default=None,
        help="Optional Sentinel-2 RGB GeoTIFF for side-by-side display.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="Output directory for the figure.",
    )
    parser.add_argument("--seed", type=int, default=42, help="PCA random seed.")
    parser.add_argument(
        "--stretch",
        choices=["percentile", "fixed"],
        default="percentile",
        help="RGB stretch mode: 'percentile' (2/98) or 'fixed' ([0, 0.25] reflectance).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return parser.parse_args()


def main() -> None:
    """Compute PCA false-color image and save the figure."""
    args = parse_args()
    result = compute_pca_rgb(args.embed, args.seed)

    s2_rgb = None
    if args.rgb is not None:
        s2_rgb = _load_s2_rgb(args.rgb, (result.height, result.width), args.stretch)

    fig = make_pca_figure(result, s2_rgb)

    args.out.mkdir(parents=True, exist_ok=True)
    fig_path = args.out / "pca_false_color.png"
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
