"""OlmoEarth Studio API client for computing custom embeddings.

Wraps the Studio REST API with helpers for project creation, embeddings model
setup, prediction submission, polling, and result download. Only the optional
custom-embeddings example uses it; the main examples read the public global
embeddings instead.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.merge
import rasterio.windows
import requests

from olmoearth_embeddings_tutorial.common.constants import EMBEDDINGS_FILENAME
from olmoearth_embeddings_tutorial.common.embedding_utils import NODATA_EMB

DEFAULT_BASE_URL = "https://olmoearth.allenai.org"


class StudioClient:
    """Thin wrapper around the OlmoEarth Studio REST API.

    Args:
        api_key: Bearer token for authentication.
        base_url: Studio API root (no trailing slash).
    """

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL) -> None:
        """Initialize the client with an API key and optional base URL."""
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._base = base_url.rstrip("/")

    @staticmethod
    def _first_record(resp: requests.Response) -> dict[str, Any]:
        """Raise on HTTP errors and return the first record of the response.

        Studio wraps responses as ``{"records": [...], "meta": ..., "errors": ...}``.
        """
        if not resp.ok:
            raise RuntimeError(
                f"{resp.request.method} {resp.url} failed with "
                f"{resp.status_code}: {resp.text}"
            )
        records = resp.json().get("records") or []
        if not records:
            raise RuntimeError(f"{resp.request.method} {resp.url} returned no records")
        return records[0]  # type: ignore[no-any-return]

    def get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Issue a GET request and return the first response record."""
        resp = self._session.get(f"{self._base}{path}", **kwargs)
        return self._first_record(resp)

    def post(self, path: str, json_body: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Issue a POST request and return the first response record."""
        resp = self._session.post(f"{self._base}{path}", json=json_body, **kwargs)
        return self._first_record(resp)

    def create_project(
        self, name: str, organization_id: str | None = None
    ) -> dict[str, Any]:
        """Create a new project and return its metadata.

        *organization_id* is required when the caller belongs to more than one
        organization.
        """
        body: dict[str, Any] = {"name": name}
        if organization_id is not None:
            body["organization_id"] = organization_id
        return self.post("/api/v1/projects", body)

    def create_embeddings_model(
        self,
        project_id: str,
        model_name: str,
        foundation_model_version: str = "v1.3",
        resolution: str = "forty_meter",
        num_periods: int = 12,
        imagery_sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an embeddings model via the wizard endpoint.

        The API defaults to an older foundation model version, so the version
        is always sent explicitly. v1.3 has only a Base encoder, which is the
        API's default variant.
        """
        if imagery_sources is None:
            imagery_sources = ["sentinel2_l2a"]
        body: dict[str, Any] = {
            "model_type": "embeddings",
            "wizard_answers": {
                "wizard_id": "embeddings_v1",
                "project_id": project_id,
                "model_name": model_name,
                "foundation_model_version": foundation_model_version,
                "resolution": resolution,
                "num_periods": num_periods,
                "imagery_sources": imagery_sources,
            },
        }
        return self.post("/api/v1/models/from_config", body)

    def create_prediction(
        self,
        project_id: str,
        model_id: str,
        name: str,
        geojson: dict[str, Any],
        datetime_range: str,
    ) -> dict[str, Any]:
        """Submit a new prediction job.

        *datetime_range* is ``"start/end"`` in ISO 8601 and applies to every
        feature in *geojson*. Times without a timezone are taken as UTC.
        """
        start_time, end_time = (
            _to_utc_iso(value) for value in datetime_range.split("/")
        )
        body: dict[str, Any] = {
            "project_id": project_id,
            "model_id": model_id,
            "name": name,
            "geojson": geojson,
            "start_time": start_time,
            "end_time": end_time,
        }
        return self.post("/api/v1/predictions", body)

    def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        """Fetch current prediction status."""
        return self.get(f"/api/v1/predictions/{prediction_id}")

    def poll_prediction(
        self,
        prediction_id: str,
        poll_interval: float = 60.0,
        max_wait: float = 7200.0,
    ) -> dict[str, Any]:
        """Block until the prediction reaches a terminal status.

        Polls every *poll_interval* seconds (default 60 s). Dropped
        connections are retried on the next poll, since the server may close
        idle keep-alive connections between polls.

        Raises:
            RuntimeError: If the prediction fails or times out.
        """
        elapsed = 0.0
        while elapsed < max_wait:
            try:
                pred = self.get_prediction(prediction_id)
            except requests.ConnectionError as exc:
                print(
                    f"  prediction {prediction_id}: connection error, retrying ({exc})"
                )
                time.sleep(poll_interval)
                elapsed += poll_interval
                continue
            status = pred.get("status", "unknown")
            print(f"  prediction {prediction_id}: {status} ({elapsed:.0f}s)")
            if status == "completed":
                return pred
            if status in ("failed", "cancelled", "result_creation_failed"):
                raise RuntimeError(
                    f"Prediction {prediction_id} ended with status '{status}': "
                    f"{pred.get('error_message')}"
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(
            f"Prediction {prediction_id} still running after {max_wait:.0f}s"
        )

    def download_prediction_result(
        self,
        prediction: dict[str, Any],
        out_dir: Path,
        filename: str = EMBEDDINGS_FILENAME,
    ) -> Path:
        """Download a completed prediction's embeddings as one GeoTIFF.

        Larger areas come back as several tiles, so every .tif in the result
        ZIP is mosaicked into a single raster, saved as *filename* inside
        *out_dir*, and cropped to its valid (non-nodata) extent.

        Returns:
            Path to the saved .tif file.
        """
        result = prediction.get("result")
        if not result:
            raise RuntimeError(f"Prediction {prediction['id']} has no result")
        url = f"{self._base}/api/v1/prediction-results/files"
        resp = self._session.get(
            url, params={"download_token": result["download_token"]}
        )
        resp.raise_for_status()

        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / filename
        with (
            zipfile.ZipFile(io.BytesIO(resp.content)) as zf,
            tempfile.TemporaryDirectory() as tmp,
        ):
            tile_paths: list[Path] = []
            for name in zf.namelist():
                if name.endswith(".tif"):
                    tile_path = Path(tmp) / f"tile_{len(tile_paths)}.tif"
                    tile_path.write_bytes(zf.read(name))
                    tile_paths.append(tile_path)
            if not tile_paths:
                raise RuntimeError("No .tif found in prediction result ZIP")
            mosaic_tiles(tile_paths, dest)
        crop_to_valid(dest)
        print(f"  saved {dest} ({len(tile_paths)} tile(s))")
        return dest


def mosaic_tiles(tile_paths: list[Path], dest: Path) -> None:
    """Merge embedding GeoTIFF tiles that share a CRS into one GeoTIFF."""
    sources = [rasterio.open(path) for path in tile_paths]
    try:
        crs_values = {src.crs for src in sources}
        if len(crs_values) > 1:
            raise RuntimeError(
                f"Prediction result tiles use different CRSs: {crs_values}"
            )
        data, transform = rasterio.merge.merge(sources, nodata=NODATA_EMB)
        profile = sources[0].profile
    finally:
        for src in sources:
            src.close()
    profile.update(
        driver="GTiff",
        interleave="band",
        height=data.shape[1],
        width=data.shape[2],
        transform=transform,
        nodata=NODATA_EMB,
    )
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(data)


def crop_to_valid(path: Path) -> None:
    """Crop an embedding GeoTIFF in place to the bounding box of its valid pixels.

    Studio pads the output raster beyond the requested area with nodata, so
    cropping keeps the figures (and the imagery downloaded onto this grid)
    focused on the area of interest.
    """
    with rasterio.open(path) as src:
        data = src.read()
        rows, cols = np.nonzero(np.any(data != NODATA_EMB, axis=0))
        if rows.size == 0:
            return
        row_slice = slice(rows.min(), rows.max() + 1)
        col_slice = slice(cols.min(), cols.max() + 1)
        window = rasterio.windows.Window.from_slices(row_slice, col_slice)
        profile = src.profile
        profile.update(
            driver="GTiff",
            interleave="band",
            width=int(window.width),
            height=int(window.height),
            transform=src.window_transform(window),
        )
        cropped = data[:, row_slice, col_slice]
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(cropped)


def _to_utc_iso(value: str) -> str:
    """Return an ISO 8601 timestamp, assuming UTC when no timezone is given."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.isoformat()


def client_from_args(
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> StudioClient:
    """Build a StudioClient from an explicit key or the environment.

    Falls back to the ``OLMOEARTH_API_KEY`` environment variable when
    *api_key* is ``None``.

    Raises:
        SystemExit: If no API key is available.
    """
    key = api_key or os.environ.get("OLMOEARTH_API_KEY")
    if not key:
        raise SystemExit(
            "Provide --api-key or set the OLMOEARTH_API_KEY environment variable."
        )
    return StudioClient(key, base_url)
