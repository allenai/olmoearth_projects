"""One-time setup: create a Studio project and embeddings models.

Creates a project and two OlmoEarth-v1.3-Base embeddings models:
  1. Annual (12 monthly periods) -- used by similarity, segmentation, and PCA.
  2. Monthly (1 period) -- used by change detection.

Both use 40-meter resolution with Sentinel-2 L2A imagery.  Writes the
resulting IDs to a JSON config file that the per-example compute scripts read.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.setup_model --api-key $OLMOEARTH_API_KEY
"""

from __future__ import annotations

import argparse
from pathlib import Path

from olmoearth_embeddings_tutorial.common.studio_client import (
    client_from_args,
    save_config,
)

DEFAULT_CONFIG = Path("config.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "OlmoEarth Studio API key. Falls back to the "
            "OLMOEARTH_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="https://olmoearth.allenai.org",
        help="Studio API base URL.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to write the JSON config file. Default: %(default)s.",
    )
    parser.add_argument(
        "--project-name",
        default="embeddings-tutorial",
        help="Name for the new Studio project.",
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
        "--model-name",
        default="blog-embeddings-v1-3-40m-s2",
        help="Name for the embeddings model.",
    )
    parser.add_argument(
        "--foundation-model-version",
        default="v1.3",
        help="OlmoEarth foundation model version. Default: %(default)s.",
    )
    parser.add_argument(
        "--resolution",
        default="forty_meter",
        help="Spatial resolution identifier. Default: %(default)s.",
    )
    parser.add_argument(
        "--num-periods",
        type=int,
        default=12,
        help="Number of monthly periods (annual composite). Default: %(default)s.",
    )
    return parser.parse_args()


def main() -> None:
    """Create a Studio project and two embeddings models, then save IDs."""
    args = parse_args()
    client = client_from_args(args.api_key, args.base_url)

    print(f"Creating project '{args.project_name}'...")
    project = client.create_project(args.project_name, args.organization_id)
    project_id = project["id"]
    print(f"  project_id: {project_id}")

    annual_name = f"{args.model_name}-annual"
    print(f"Creating annual embeddings model '{annual_name}'...")
    annual_model = client.create_embeddings_model(
        project_id=project_id,
        model_name=annual_name,
        foundation_model_version=args.foundation_model_version,
        resolution=args.resolution,
        num_periods=args.num_periods,
        imagery_sources=["sentinel2_l2a"],
    )
    model_id = annual_model["id"]
    print(f"  model_id: {model_id}")

    monthly_name = f"{args.model_name}-monthly"
    print(f"Creating monthly embeddings model '{monthly_name}'...")
    monthly_model = client.create_embeddings_model(
        project_id=project_id,
        model_name=monthly_name,
        foundation_model_version=args.foundation_model_version,
        resolution=args.resolution,
        num_periods=1,
        imagery_sources=["sentinel2_l2a"],
    )
    monthly_model_id = monthly_model["id"]
    print(f"  monthly_model_id: {monthly_model_id}")

    save_config(
        args.config,
        {
            "project_id": project_id,
            "model_id": model_id,
            "monthly_model_id": monthly_model_id,
            "foundation_model_version": args.foundation_model_version,
            "resolution": args.resolution,
            "num_periods": args.num_periods,
        },
    )
    print("Setup complete.")


if __name__ == "__main__":
    main()
