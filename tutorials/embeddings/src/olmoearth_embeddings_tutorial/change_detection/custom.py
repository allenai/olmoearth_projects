"""Compute custom monthly embeddings for change detection with OlmoEarth Studio.

The global embeddings are annual composites. With Studio you can compute
embeddings for any time window, resolution, and imagery sources. This script
creates a one-month OlmoEarth v1.3 embeddings model (40 meters, Sentinel-2),
computes embeddings for September 2023 (before the Park Fire) and September
2024 (just after it), and downloads them with a September Sentinel-2 RGB
composite for each month. Requires an OlmoEarth Studio API key.

Example:
    $ export OLMOEARTH_API_KEY="your-key-here"
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.custom
"""

from __future__ import annotations

import argparse
from pathlib import Path

from olmoearth_embeddings_tutorial.change_detection.download import (
    PARK_FIRE_GEOJSON,
)
from olmoearth_embeddings_tutorial.common.constants import S2_RGB_FILENAME
from olmoearth_embeddings_tutorial.common.imagery_sources import download_s2_rgb
from olmoearth_embeddings_tutorial.common.studio_client import (
    DEFAULT_BASE_URL,
    client_from_args,
)

MONTHS = {
    "sept_2023": "2023-09-01/2023-10-01",
    "sept_2024": "2024-09-01/2024-10-01",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key",
        default=None,
        help="Studio API key. Falls back to the OLMOEARTH_API_KEY environment variable.",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="Studio API base URL."
    )
    parser.add_argument(
        "--organization-id",
        default=None,
        help=(
            "Organization to create the project in. Required if you belong "
            "to more than one organization."
        ),
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Existing project to use instead of creating one.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Existing monthly embeddings model to use instead of creating one.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/change_detection/custom"),
        help="Root directory for downloaded data.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the model if needed, compute both months, and download data."""
    args = parse_args()
    client = client_from_args(args.api_key, args.base_url)

    project_id = args.project_id
    if project_id is None:
        project = client.create_project("embeddings-tutorial", args.organization_id)
        project_id = project["id"]
        print(f"Created project {project_id}")

    model_id = args.model_id
    if model_id is None:
        model = client.create_embeddings_model(
            project_id=project_id,
            model_name="park-fire-monthly-40m-s2",
            num_periods=1,
        )
        model_id = model["id"]
        print(f"Created monthly embeddings model {model_id}")
        print(f"  To reuse it: --project-id {project_id} --model-id {model_id}")

    predictions = {}
    for label, datetime_range in MONTHS.items():
        pred = client.create_prediction(
            project_id=project_id,
            model_id=model_id,
            name=f"park-fire-{label}",
            geojson=PARK_FIRE_GEOJSON,
            datetime_range=datetime_range,
        )
        predictions[label] = pred["id"]
        print(f"Submitted {label}: prediction {pred['id']}")

    print("Waiting for predictions (this usually takes 30 to 60 minutes)...")
    for label, prediction_id in predictions.items():
        month_dir = args.out_dir / label
        completed = client.poll_prediction(prediction_id)
        embed_path = client.download_prediction_result(completed, month_dir)
        print(f"Downloading Sentinel-2 RGB for {label}...")
        download_s2_rgb(
            embed_path,
            month_dir / S2_RGB_FILENAME,
            datetime_range=MONTHS[label],
        )

    print(f"\nDone. Data saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
