"""Compute and download data for the few-shot segmentation example.

Submits a prediction for Ca Mau (Vietnam), polls until complete, then
downloads the embedding COG, Sentinel-2 RGB composite, and ESA WorldCover.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.segmentation.compute --config config.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from olmoearth_embeddings_tutorial.common.constants import (
    S2_RGB_FILENAME,
    WORLDCOVER_FILENAME,
)
from olmoearth_embeddings_tutorial.common.imagery_sources import (
    download_s2_rgb,
    download_worldcover,
)
from olmoearth_embeddings_tutorial.common.studio_client import (
    StudioClient,
    client_from_args,
    load_config,
)

CA_MAU_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [104.72, 8.53],
                        [104.72, 8.93],
                        [105.12, 8.93],
                        [105.12, 8.53],
                        [104.72, 8.53],
                    ]
                ],
            },
        }
    ],
}

DATETIME_RANGE = "2024-01-01/2024-12-31"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the config file written by setup_model.py.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Studio API key (or set OLMOEARTH_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default="https://olmoearth.allenai.org",
        help="Studio API base URL.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/segmentation"),
        help="Root directory for downloaded data.",
    )
    return parser.parse_args()


def _submit_and_download(
    client: StudioClient,
    project_id: str,
    model_id: str,
    name: str,
    geojson: dict,  # type: ignore[type-arg]
    out_dir: Path,
) -> Path:
    """Submit a prediction, poll, and download the embedding COG."""
    print(f"Submitting prediction: {name}...")
    pred = client.create_prediction(
        project_id=project_id,
        model_id=model_id,
        name=name,
        geojson=geojson,
        datetime_range=DATETIME_RANGE,
    )
    prediction_id = pred["id"]
    print(f"  prediction_id: {prediction_id}")

    print("Polling for completion...")
    result = client.poll_prediction(prediction_id)
    prediction_result = result.get("result")
    if not prediction_result:
        raise RuntimeError(f"No results for prediction {prediction_id}")
    download_token = prediction_result["download_token"]

    print("Downloading embedding COG...")
    return client.download_prediction_result(download_token, out_dir)


def main() -> None:
    """Submit prediction and download all data for Ca Mau."""
    args = parse_args()
    cfg = load_config(args.config)
    client = client_from_args(args.api_key, args.base_url)

    project_id = cfg["project_id"]
    model_id = cfg["model_id"]

    ca_mau_dir = args.out_dir / "ca_mau"

    ca_mau_embed = _submit_and_download(
        client, project_id, model_id, "ca-mau-embeddings", CA_MAU_GEOJSON, ca_mau_dir
    )

    print("\nDownloading Sentinel-2 RGB...")
    download_s2_rgb(
        ca_mau_embed,
        ca_mau_dir / S2_RGB_FILENAME,
        datetime_range="2024-01-01/2024-12-31",
    )

    print("Downloading WorldCover...")
    download_worldcover(ca_mau_embed, ca_mau_dir / WORLDCOVER_FILENAME)

    print(f"\nDone. Data saved under: {ca_mau_dir}")


if __name__ == "__main__":
    main()
