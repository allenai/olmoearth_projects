"""Embedding loading utilities for OlmoEarth tutorial scripts.

Provides a consistent interface for reading OlmoEarth embedding GeoTIFFs and
identifying valid (non-nodata) pixels.

Embeddings are stored as int8 with the same signed-power quantization as the
OlmoEarth v1.3 global embeddings on Source Cooperative:
``value = sign(q) * (|q| / 127.5) ** 2``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

NODATA_EMB: int = -128
"""Sentinel value used in int8 embedding rasters for missing data."""

QUANTIZATION_SCALE = 127.5
"""Divisor in the signed-power quantization formula."""


def dequantize(quantized: np.ndarray) -> np.ndarray:
    """Convert int8 quantized embeddings to float32 vectors."""
    scaled = quantized.astype(np.float32) / QUANTIZATION_SCALE
    return np.sign(scaled) * scaled**2


def quantize(vectors: np.ndarray) -> np.ndarray:
    """Convert float32 embedding vectors to int8 (the inverse of :func:`dequantize`)."""
    scaled = np.sign(vectors) * np.sqrt(np.abs(vectors)) * QUANTIZATION_SCALE
    return np.clip(np.round(scaled), -127, 127).astype(np.int8)


def load_embeddings(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, rasterio.DatasetReader]:
    """Load an OlmoEarth embedding GeoTIFF.

    Args:
        path: Path to a multi-band int8 GeoTIFF written by the download scripts.

    Returns:
        A 3-tuple of:
        - ``embeddings``: dequantized float32 array of shape ``(C, H, W)``.
        - ``valid``: boolean mask of shape ``(H, W)`` where ``True`` means all
          bands have real data (not the nodata sentinel).
        - ``dataset``: the open :class:`rasterio.DatasetReader`. The caller is
          responsible for closing it when done.
    """
    dataset = rasterio.open(path)
    quantized = dataset.read()
    valid = np.all(quantized != NODATA_EMB, axis=0)
    embeddings = dequantize(quantized)
    return embeddings, valid, dataset
