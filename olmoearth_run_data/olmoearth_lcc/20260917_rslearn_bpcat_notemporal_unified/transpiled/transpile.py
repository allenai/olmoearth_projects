"""Write what olmoearth_run transpiles ../olmoearth_config.yaml into for prediction.

Run from an olmoearth_run checkout (so its code and lib/shared are importable):

    PYTHONPATH=src:lib/shared/src python <this dir>/transpile.py

The files here are generated for review; olmoearth_run makes them itself at prediction time.
"""

import json
from pathlib import Path

import yaml
from olmoearth_run.runner.models.operational_context import (
    DatasetOperationalContext,
    ModelOperationalContext,
)
from olmoearth_run.runner.tools.olmoearth_config.transpilers.olmoearth_to_rslearn import (
    to_dataset_config,
    to_model_config,
)
from olmoearth_run.shared.models.api.step_type import StepType
from olmoearth_run.shared.models.model_stage_paths import ModelStagePaths
from olmoearth_run.shared.tools.inference_results_config_utils import (
    convert_output_to_inference_results_config,
)
from olmoearth_shared.models.olmoearth_config.olmoearth_config import OlmoEarthConfig

HERE = Path(__file__).resolve().parent
# Placeholders for the paths olmoearth_run fills in per prediction.
DATASET_PATH = "<dataset_path>"
MODEL_STAGE_ROOT = "<model_stage_root>"


def main() -> None:
    """Transpile the unified config and write the rslearn configs next to this script."""
    config = OlmoEarthConfig.from_yaml(
        (HERE.parent / "olmoearth_config.yaml").read_text()
    )

    dataset = to_dataset_config(
        config, DatasetOperationalContext(dataset_path=DATASET_PATH)
    )
    (HERE / "dataset_config.json").write_text(
        json.dumps(
            dataset.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True
        )
        + "\n"
    )

    ops = ModelOperationalContext(
        step_type=StepType.RUN_INFERENCE,
        dataset_path=DATASET_PATH,
        num_data_worker_processes=4,
        model_stage_paths=ModelStagePaths(root_path=MODEL_STAGE_ROOT),
    )
    model = to_model_config(config, ops)
    (HERE / "model.yaml").write_text(
        yaml.safe_dump(
            model.model_dump(mode="json", exclude_none=True), sort_keys=False, width=120
        )
    )

    assert config.data.output is not None
    results = convert_output_to_inference_results_config(config.data.output)
    (HERE / "inference_results_config.json").write_text(
        json.dumps(
            results.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
