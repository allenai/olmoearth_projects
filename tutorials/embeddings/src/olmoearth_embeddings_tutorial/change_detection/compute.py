"""Compute and download data for the change detection example.

Submits two predictions for the Park Fire region (Butte County, California)
using monthly embeddings -- September 2023 (before) and September 2024
(after) -- then downloads embedding COGs and Sentinel-2 RGB composites for
both periods.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.compute --config config.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from olmoearth_embeddings_tutorial.common.constants import S2_RGB_FILENAME
from olmoearth_embeddings_tutorial.common.imagery_sources import download_s2_rgb
from olmoearth_embeddings_tutorial.common.studio_client import (
    StudioClient,
    client_from_args,
    load_config,
)

PARK_FIRE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-121.85, 39.70],
                        [-121.85, 39.85],
                        [-121.68, 39.85],
                        [-121.68, 39.70],
                        [-121.85, 39.70],
                    ]
                ],
            },
        }
    ],
}

BEFORE_DATETIME = "2023-09-01/2023-10-01"
AFTER_DATETIME = "2024-09-01/2024-10-01"


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
        default=Path("data/change_detection"),
        help="Root directory for downloaded data.",
    )
    return parser.parse_args()


def _submit(
    client: StudioClient,
    project_id: str,
    model_id: str,
    name: str,
    geojson: dict,  # type: ignore[type-arg]
    datetime_range: str,
) -> str:
    """Submit a prediction and return its ID."""
    print(f"Submitting prediction: {name}...")
    pred = client.create_prediction(
        project_id=project_id,
        model_id=model_id,
        name=name,
        geojson=geojson,
        datetime_range=datetime_range,
    )
    prediction_id = pred["id"]
    print(f"  prediction_id: {prediction_id}")
    return prediction_id


def _poll_and_download(
    client: StudioClient,
    prediction_id: str,
    out_dir: Path,
) -> Path:
    """Poll a prediction until complete, then download the embedding COG."""
    print(f"Polling prediction {prediction_id}...")
    result = client.poll_prediction(prediction_id)
    prediction_result = result.get("result")
    if not prediction_result:
        raise RuntimeError(f"No results for prediction {prediction_id}")
    download_token = prediction_result["download_token"]

    print("Downloading embedding COG...")
    return client.download_prediction_result(download_token, out_dir)


def main() -> None:
    """Submit both predictions and download all data."""
    args = parse_args()
    cfg = load_config(args.config)
    client = client_from_args(args.api_key, args.base_url)

    project_id = cfg["project_id"]
    monthly_model_id = cfg.get("monthly_model_id")
    if not monthly_model_id:
        raise SystemExit(
            "config.json is missing 'monthly_model_id'. "
            "Re-run setup_model.py to create both annual and monthly models."
        )

    before_dir = args.out_dir / "sept_2023"
    after_dir = args.out_dir / "sept_2024"

    before_id = _submit(
        client,
        project_id,
        monthly_model_id,
        "park-fire-sept-2023",
        PARK_FIRE_GEOJSON,
        BEFORE_DATETIME,
    )
    after_id = _submit(
        client,
        project_id,
        monthly_model_id,
        "park-fire-sept-2024",
        PARK_FIRE_GEOJSON,
        AFTER_DATETIME,
    )

    before_embed = _poll_and_download(client, before_id, before_dir)
    after_embed = _poll_and_download(client, after_id, after_dir)

    print("\nDownloading Sentinel-2 RGB for September 2023...")
    download_s2_rgb(
        before_embed,
        before_dir / S2_RGB_FILENAME,
        datetime_range="2023-09-01/2023-10-01",
    )

    print("Downloading Sentinel-2 RGB for September 2024...")
    download_s2_rgb(
        after_embed, after_dir / S2_RGB_FILENAME, datetime_range="2024-09-01/2024-10-01"
    )

    print(f"\nDone. Data saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
