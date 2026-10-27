"""Sentinel-2 RGB stretching utilities.

Provides two modes for mapping float32 reflectance values to display [0, 1]:

* **percentile** (default): scene-adaptive 2nd/98th percentile stretch.
* **fixed**: linear stretch mapping reflectance [0, 0.25] to [0, 1].
  Equivalent to mapping DN [0, 2500] after the Sentinel-2 L2A BOA offset
  correction. Produces consistent brightness across scenes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

StretchMode = Literal["percentile", "fixed"]

FIXED_MAX_REFLECTANCE = 0.25
NODATA_FILL = 0.15


def stretch_rgb(
    rgb_hwc: np.ndarray,
    mode: StretchMode = "percentile",
    gamma: float = 1.0,
) -> np.ndarray:
    """Stretch a (H, W, 3) float32 reflectance array to [0, 1] for display.

    Args:
        rgb_hwc: (H, W, 3) float32 array in reflectance units.
        mode: ``"percentile"`` for scene-adaptive 2/98 stretch, or
            ``"fixed"`` for a fixed [0, 0.25] reflectance range.
        gamma: Optional gamma correction (applied after stretching).
            Values < 1 brighten; > 1 darken.  Default 1.0 (no correction).

    Returns:
        (H, W, 3) float32 array clipped to [0, 1] with nodata filled to
        ``NODATA_FILL``.
    """
    out = rgb_hwc.astype(np.float32).copy()
    nans = np.isnan(out).any(axis=-1) | (out.sum(axis=-1) == 0)
    if nans.all():
        return np.full_like(out, NODATA_FILL)

    if mode == "fixed":
        out = np.clip(out / FIXED_MAX_REFLECTANCE, 0.0, 1.0)
    else:
        lo, hi = np.nanpercentile(out[~nans], [2, 98])
        out = np.clip((out - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)

    if abs(gamma - 1.0) > 1e-6:
        out = np.power(out, gamma)

    out[nans] = NODATA_FILL
    return out
