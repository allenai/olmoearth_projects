"""Compute and download embeddings + Sentinel-2 RGB for the similarity example.

Submits a prediction over California's Central Valley, polls until complete,
downloads the embedding COG, and fetches a Sentinel-2 L2A median composite
aligned to the same grid.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.similarity.compute --config config.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from olmoearth_embeddings_tutorial.common.constants import S2_RGB_FILENAME
from olmoearth_embeddings_tutorial.common.imagery_sources import download_s2_rgb
from olmoearth_embeddings_tutorial.common.studio_client import (
    client_from_args,
    load_config,
)

CENTRAL_VALLEY_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-120.75, 36.65],
                        [-120.75, 37.15],
                        [-120.15, 37.15],
                        [-120.15, 36.65],
                        [-120.75, 36.65],
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
        default=Path("data/central_valley"),
        help="Directory for downloaded data.",
    )
    return parser.parse_args()


def main() -> None:
    """Submit prediction, download embeddings and S2 RGB."""
    args = parse_args()
    cfg = load_config(args.config)
    client = client_from_args(args.api_key, args.base_url)

    project_id = cfg["project_id"]
    model_id = cfg["model_id"]

    print("Submitting Central Valley prediction...")
    pred = client.create_prediction(
        project_id=project_id,
        model_id=model_id,
        name="central-valley-embeddings",
        geojson=CENTRAL_VALLEY_GEOJSON,
        datetime_range=DATETIME_RANGE,
    )
    prediction_id = pred["id"]
    print(f"  prediction_id: {prediction_id}")

    print("Polling for completion...")
    result = client.poll_prediction(prediction_id)

    prediction_result = result.get("result")
    if not prediction_result:
        raise RuntimeError("No prediction results returned.")
    download_token = prediction_result["download_token"]

    print("Downloading embedding COG...")
    embed_path = client.download_prediction_result(download_token, args.out_dir)

    print("Downloading Sentinel-2 RGB median composite...")
    s2_path = args.out_dir / S2_RGB_FILENAME
    download_s2_rgb(embed_path, s2_path, datetime_range="2024-01-01/2024-12-31")

    print(f"Done. Embeddings: {embed_path}, S2 RGB: {s2_path}")


if __name__ == "__main__":
    main()
