"""Register the unified LCC config in Studio, beside the legacy LCC model.

Takes the project, foundation model and checkpoint from an existing (legacy) LCC model's
ingredients, validates olmoearth_config.yaml against the server, and registers a new
model with it. Olmoearth_run copies the checkpoint into the new model's stage, so the
legacy model is unchanged. Registering needs a platform-admin API key.

Dry run (default), then register:

    export API_KEY=<Studio API key>
    python register_in_studio.py --api-url https://staging.olmoearth.allenai.org --from-model <legacy model id>
    python register_in_studio.py --api-url https://staging.olmoearth.allenai.org --from-model <legacy model id> --submit

Without a legacy model to copy from (e.g. on staging), pass --project-id, --foundation-model
and --checkpoint instead of --from-model.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
import yaml

HERE = Path(__file__).resolve().parent
# Deploys point this olmoearth_run container image at the newest runner build.
CONTAINER_IMAGE = "runner:latest"
CHECKPOINT = "gs://ai2-rslearn-projects-data/projects/2026_09_17_lcc/rslearn_bpcat_notemporal/best.ckpt"


def api(session: requests.Session, method: str, url: str, **kwargs: Any) -> Any:
    """Call the Studio API and return its records, failing loudly on an error."""
    response = session.request(method, url, timeout=120, **kwargs)
    if not response.ok:
        sys.exit(f"{method} {url} -> {response.status_code}: {response.text}")
    return response.json().get("records")


def main() -> None:
    """Validate the unified config and, with --submit, register it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="Studio base URL, e.g. https://staging.olmoearth.allenai.org",
    )
    parser.add_argument(
        "--from-model", help="ID of the legacy LCC model to take ingredients from"
    )
    parser.add_argument(
        "--project-id", help="Project to register in (default: the --from-model's)"
    )
    parser.add_argument(
        "--foundation-model",
        help="olmoearth_run foundation model name (default: the --from-model's)",
    )
    parser.add_argument(
        "--checkpoint",
        help=f"Checkpoint path (default: the --from-model's, else {CHECKPOINT})",
    )
    parser.add_argument(
        "--name",
        default="OlmoEarth LCC (unified config)",
        help="Name for the new model",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Register the model (otherwise only validate)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY")
    if not api_key:
        sys.exit("Set API_KEY to a Studio API key")
    base = args.api_url.rstrip("/") + "/api/v1"
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"

    ingredients: dict[str, Any] = {}
    if args.from_model:
        [ingredients] = api(
            session, "GET", f"{base}/models/{args.from_model}/inspect_ingredients"
        )
    project_id = args.project_id or ingredients.get("project_id")
    foundation_model = args.foundation_model or ingredients.get("foundation_model_name")
    checkpoint = args.checkpoint or ingredients.get("checkpoint_path") or CHECKPOINT
    if not project_id or not foundation_model:
        sys.exit("Pass --from-model, or both --project-id and --foundation-model")

    config_yaml = (HERE / "olmoearth_config.yaml").read_text()
    api(
        session,
        "POST",
        f"{base}/models/validate_config",
        json={"config_yaml": config_yaml},
    )
    print("config is valid on the server")

    registration = {
        "name": args.name,
        "model_type": "fine_tuned",
        "project_id": project_id,
        "olmoearth_config": yaml.safe_load(config_yaml),
        "container_image_name": CONTAINER_IMAGE,
        "foundation_model_name": foundation_model,
        "checkpoint_path": checkpoint,
    }
    summary = {k: v for k, v in registration.items() if k != "olmoearth_config"}
    print(json.dumps(summary, indent=2))
    if not args.submit:
        print("dry run: pass --submit to register")
        return

    [model] = api(session, "POST", f"{base}/models/register", json=registration)
    print(f"registered model {model['id']} ({model['name']})")


if __name__ == "__main__":
    main()
