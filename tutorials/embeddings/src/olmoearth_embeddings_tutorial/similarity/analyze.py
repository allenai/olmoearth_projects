"""Similarity analysis: cosine-similarity heatmap and patch mosaic.

Loads the Central Valley embedding COG and Sentinel-2 RGB, picks a query
pixel, computes a per-pixel cosine similarity map, and generates two figures:

1. A similarity heatmap overlaid on S2 imagery.
2. A patch mosaic showing the most and least similar locations.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.similarity.analyze \
        --embed data/central_valley/embeddings.tif \
        --rgb data/central_valley/s2_rgb.tif \
        --out figures/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.fill
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from rasterio.transform import rowcol
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as warp_transform

from olmoearth_embeddings_tutorial.common.embedding_utils import load_embeddings
from olmoearth_embeddings_tutorial.common.rgb_stretch import StretchMode, stretch_rgb

QUERY_LON = -120.45
QUERY_LAT = 36.90
PATCH_SIZE = 95
K = 3
QUERY_HALF = 3

COLOR_QUERY = "#56B4E9"
COLOR_HIGH = "#facc15"
COLOR_LOW = "#b91c1c"


@dataclass(frozen=True)
class SimilarityResult:
    """Intermediate results from :func:`compute_similarity`."""

    sim: np.ndarray
    embeddings: np.ndarray
    valid: np.ndarray
    rgb_raw: np.ndarray
    rgb_ok: np.ndarray
    cy: int
    cx: int
    stretch: StretchMode = "percentile"


def _cosine_similarity_map(
    embeddings: np.ndarray,
    valid: np.ndarray,
    query_mask: np.ndarray,
) -> np.ndarray:
    """Compute per-pixel cosine similarity to the mean query embedding."""
    c, h, w = embeddings.shape
    q = embeddings[:, query_mask].mean(axis=1)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    flat = embeddings.reshape(c, h * w)
    norms = np.maximum(np.linalg.norm(flat, axis=0, keepdims=True), 1e-8)
    sim = (q @ (flat / norms)).reshape(h, w)
    return np.where(valid, sim, np.nan)


def _lonlat_to_rowcol(
    dataset: rasterio.DatasetReader,
    lon: float,
    lat: float,
) -> tuple[int, int]:
    """Convert WGS 84 lon/lat to pixel (row, col) on *dataset*'s grid."""
    xs, ys = warp_transform("EPSG:4326", dataset.crs, [lon], [lat])
    rc = rowcol(dataset.transform, xs[0], ys[0])
    return int(rc[0]), int(rc[1])


def _load_rgb(
    rgb_path: Path,
    match: rasterio.DatasetReader,
) -> np.ndarray:
    """Load a 3-band GeoTIFF, reprojecting to *match* grid if needed."""
    with rasterio.open(rgb_path) as ref:
        same = (
            ref.height == match.height
            and ref.width == match.width
            and ref.crs == match.crs
        )
        if same:
            return ref.read([1, 2, 3]).astype(np.float32)
        out = np.zeros((3, match.height, match.width), dtype=np.float32)
        for b in range(1, 4):
            reproject(
                source=rasterio.band(ref, b),
                destination=out[b - 1],
                src_transform=ref.transform,
                src_crs=ref.crs,
                dst_transform=match.transform,
                dst_crs=match.crs,
                resampling=Resampling.bilinear,
            )
        return out


def _interpolate_nodata(rgb: np.ndarray, max_dist: float = 48.0) -> np.ndarray:
    """Fill small nodata holes in a (3, H, W) float32 array."""
    out = np.asarray(rgb, dtype=np.float32).copy()
    for b in range(3):
        plane = out[b]
        valid = np.isfinite(plane)
        if not np.any(valid) or np.all(valid):
            continue
        mask = valid.astype(np.uint8)
        tmp = np.where(valid, plane, 0.0).astype(np.float32)
        out[b] = rasterio.fill.fillnodata(
            tmp, mask=mask, max_search_distance=max_dist, smoothing_iterations=1
        )
    return out


def _global_stretch(
    rgb_chw: np.ndarray,
    gamma: float = 0.92,
    mode: StretchMode = "percentile",
) -> np.ndarray:
    """Stretch (3, H, W) to (H, W, 3) in [0, 1]."""
    hwc = np.moveaxis(rgb_chw, 0, -1)
    return stretch_rgb(hwc, mode=mode, gamma=gamma)


def _patch_stretch(
    patch_chw: np.ndarray,
    gamma: float = 0.92,
    mode: StretchMode = "percentile",
) -> np.ndarray:
    """Per-patch stretch for thumbnail display."""
    hwc = np.moveaxis(np.asarray(patch_chw, dtype=np.float32), 0, -1)
    if not np.any(np.isfinite(hwc)):
        return np.full(hwc.shape, 0.45, dtype=np.float32)
    return stretch_rgb(hwc, mode=mode, gamma=gamma)


def _boxes_overlap(r1: int, c1: int, r2: int, c2: int, half: int, gap: int = 4) -> bool:
    """True if two axis-aligned square patches overlap (with buffer)."""
    return abs(r1 - r2) < 2 * half + gap and abs(c1 - c2) < 2 * half + gap


def _pick_diverse(
    sim: np.ndarray,
    valid: np.ndarray,
    margin: int,
    k: int,
    min_sep_px: float,
    ascending: bool,
    exclude: np.ndarray | None = None,
    rgb_ok: np.ndarray | None = None,
    taken: list[tuple[int, int]] | None = None,
    patch_half: int = 47,
) -> list[tuple[int, int]]:
    """Greedy selection of diverse high- or low-similarity patches."""
    h, w = sim.shape
    taken = taken or []
    candidates: list[tuple[float, int, int]] = []
    for r in range(margin, h - margin):
        for c in range(margin, w - margin):
            if not (valid[r, c] and np.isfinite(sim[r, c])):
                continue
            if rgb_ok is not None and not rgb_ok[r, c]:
                continue
            if exclude is not None and exclude[r, c]:
                continue
            candidates.append((float(sim[r, c]), r, c))
    candidates.sort(key=lambda t: t[0], reverse=not ascending)

    chosen: list[tuple[int, int]] = []
    min_sep2 = min_sep_px * min_sep_px
    for _, r, c in candidates:
        if len(chosen) >= k:
            break
        if not all((r - r2) ** 2 + (c - c2) ** 2 >= min_sep2 for r2, c2 in chosen):
            continue
        if any(_boxes_overlap(r, c, tr, tc, patch_half) for tr, tc in taken):
            continue
        chosen.append((r, c))
    return chosen[:k]


def _color_border(ax: plt.Axes, color: str, lw: float = 3.0) -> None:
    """Draw a colored border around a matplotlib Axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)


# ---------------------------------------------------------------------------
# Public API (used by both CLI and notebooks)
# ---------------------------------------------------------------------------


def compute_similarity(
    embed_path: Path,
    rgb_path: Path,
    query_lon: float = QUERY_LON,
    query_lat: float = QUERY_LAT,
    stretch: StretchMode = "percentile",
) -> SimilarityResult:
    """Load data and compute the cosine similarity map.

    Returns a :class:`SimilarityResult` with all arrays needed for figure
    generation.
    """
    print("Loading embeddings...")
    embeddings, valid, dataset = load_embeddings(embed_path)

    print("Loading Sentinel-2 RGB...")
    rgb_raw = _load_rgb(rgb_path, dataset)
    rgb_raw = _interpolate_nodata(rgb_raw)
    rgb_ok = np.all(np.isfinite(rgb_raw), axis=0)

    cy, cx = _lonlat_to_rowcol(dataset, query_lon, query_lat)
    print(f"Query pixel: row={cy}, col={cx}")

    r = QUERY_HALF
    qmask = np.zeros_like(valid, dtype=bool)
    qmask[cy - r : cy + r + 1, cx - r : cx + r + 1] = valid[
        cy - r : cy + r + 1, cx - r : cx + r + 1
    ]
    if not np.any(qmask):
        raise SystemExit("Query window has no valid embedding pixels.")

    print("Computing cosine similarity...")
    sim = _cosine_similarity_map(embeddings, valid, qmask)
    dataset.close()

    return SimilarityResult(
        sim=sim,
        embeddings=embeddings,
        valid=valid,
        rgb_raw=rgb_raw,
        rgb_ok=rgb_ok,
        cy=cy,
        cx=cx,
        stretch=stretch,
    )


def make_heatmap_figure(
    result: SimilarityResult,
) -> Figure:
    """Create the similarity heatmap figure.

    Returns the matplotlib Figure (caller decides whether to save or show).
    """
    rgb_display = _global_stretch(result.rgb_raw, mode=result.stretch)
    # Size the panels to the raster's aspect ratio, plus room for the colorbar.
    height, width = result.sim.shape
    panel_h = 6.0
    panel_w = panel_h * width / height
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * panel_w + 1.2, panel_h + 0.6), layout="constrained"
    )

    axes[0].imshow(rgb_display, interpolation="nearest")
    axes[0].plot(
        result.cx, result.cy, "x", color="red", markersize=10, markeredgewidth=2
    )
    axes[0].set_title("Sentinel-2 RGB", fontsize=13, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im = axes[1].imshow(
        result.sim, cmap="RdYlGn", vmin=-0.2, vmax=1.0, interpolation="nearest"
    )
    axes[1].plot(
        result.cx, result.cy, "x", color="black", markersize=10, markeredgewidth=2
    )
    axes[1].set_title("Cosine similarity to query", fontsize=13, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    return fig


def make_mosaic_figure(
    result: SimilarityResult,
) -> Figure:
    """Create the patch mosaic figure.

    Returns the matplotlib Figure (caller decides whether to save or show).
    """
    sim = result.sim
    valid = result.valid
    rgb_raw = result.rgb_raw
    rgb_ok = result.rgb_ok
    cy, cx = result.cy, result.cx
    _, h, w = result.embeddings.shape

    half = PATCH_SIZE // 2
    margin = half + 1
    r = QUERY_HALF

    yy, xx = np.ogrid[:h, :w]
    min_r = float(2 * half + r + 1)
    excl = (
        (yy.astype(np.float64) - cy) ** 2 + (xx.astype(np.float64) - cx) ** 2
    ) <= min_r * min_r
    min_sep = float(2 * PATCH_SIZE + 1)
    query_taken = [(cy, cx)]

    top_rc = _pick_diverse(
        sim,
        valid,
        margin,
        K,
        min_sep,
        ascending=False,
        exclude=excl,
        rgb_ok=rgb_ok,
        taken=query_taken,
        patch_half=half,
    )
    bot_rc = _pick_diverse(
        sim,
        valid,
        margin,
        K,
        min_sep,
        ascending=True,
        exclude=excl,
        rgb_ok=rgb_ok,
        taken=query_taken + list(top_rc),
        patch_half=half,
    )

    top_scores = [float(sim[r_, c_]) for r_, c_ in top_rc]
    bot_scores = [float(sim[r_, c_]) for r_, c_ in bot_rc]

    rgb_global = _global_stretch(rgb_raw, mode=result.stretch)

    def _patch(row: int, col: int) -> np.ndarray:
        crop = rgb_raw[:, row - half : row + half + 1, col - half : col + half + 1]
        return _patch_stretch(crop, mode=result.stretch)

    fig_w = 3.8 * K
    fig_h = 3.8 * 3
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        3,
        K,
        height_ratios=[1.0, 1.0, 1.0],
        left=0.02,
        right=0.98,
        top=0.92,
        bottom=0.035,
        hspace=0.12,
        wspace=0.06,
    )

    ax_ov = fig.add_subplot(gs[0, 0])
    ax_ov.imshow(rgb_global, interpolation="nearest", origin="upper")
    ax_ov.set_title("Overview", fontsize=13, fontweight="bold", pad=4)

    def _add_box(row: int, col: int, color: str, lw: float) -> None:
        x0 = col - half - 0.5
        y0 = row - half - 0.5
        ax_ov.add_patch(
            Rectangle(
                (x0, y0),
                2 * half + 1,
                2 * half + 1,
                linewidth=lw,
                edgecolor=color,
                facecolor="none",
            )
        )

    _add_box(cy, cx, COLOR_QUERY, 2.0)
    for row, col in top_rc:
        _add_box(row, col, COLOR_HIGH, 1.4)
    for row, col in bot_rc:
        _add_box(row, col, COLOR_LOW, 1.4)
    ax_ov.axis("off")

    ax_q = fig.add_subplot(gs[0, 1])
    ax_q.imshow(_patch(cy, cx), interpolation="nearest", origin="upper")
    ax_q.set_title("Query", fontsize=13, fontweight="bold", pad=4)
    _color_border(ax_q, COLOR_QUERY)

    fig.suptitle(
        "Embedding cosine similarity: query vs highest / lowest",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )

    for i in range(K):
        row, col = top_rc[i]
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(_patch(row, col), interpolation="nearest", origin="upper")
        ax.set_title(f"High {top_scores[i]:.2f}", fontsize=13, fontweight="bold", pad=4)
        _color_border(ax, COLOR_HIGH)

    for i in range(K):
        row, col = bot_rc[i]
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(_patch(row, col), interpolation="nearest", origin="upper")
        ax.set_title(f"Low {bot_scores[i]:.2f}", fontsize=13, fontweight="bold", pad=4)
        _color_border(ax, COLOR_LOW)

    fig.text(
        0.5,
        0.01,
        "Blue = query  |  Yellow = high similarity  |  Red = low similarity",
        ha="center",
        fontsize=10,
        color="0.4",
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
        required=True,
        help="Path to 3-band Sentinel-2 RGB GeoTIFF.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--query-lon",
        type=float,
        default=QUERY_LON,
        help="Query center longitude (WGS 84).",
    )
    parser.add_argument(
        "--query-lat",
        type=float,
        default=QUERY_LAT,
        help="Query center latitude (WGS 84).",
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
    """Generate similarity heatmap and patch mosaic figures."""
    args = parse_args()
    result = compute_similarity(
        args.embed, args.rgb, args.query_lon, args.query_lat, args.stretch
    )

    args.out.mkdir(parents=True, exist_ok=True)

    fig = make_heatmap_figure(result)
    heatmap_path = args.out / "similarity_heatmap.png"
    fig.savefig(heatmap_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {heatmap_path}")

    fig = make_mosaic_figure(result)
    mosaic_path = args.out / "similarity_mosaic.png"
    fig.savefig(
        mosaic_path,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=True,
    )
    plt.close(fig)
    print(f"Wrote {mosaic_path}")
    print("Done.")


if __name__ == "__main__":
    main()
