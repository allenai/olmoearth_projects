"""Start a Beaker job to run the forest loss driver inference pipeline.

This is typically executed from the Github Action:
.github/workflows/forest_loss_driver_prediction.yaml
"""

import uuid

from beaker import (
    Beaker,
    BeakerDataMount,
    BeakerDataSource,
    BeakerEnvVar,
    BeakerExperimentSpec,
    BeakerJobPriority,
)

from olmoearth_projects.utils.logging import get_logger

logger = get_logger(__name__)

WORKSPACE = "ai2/earth-systems"
BEAKER_BUDGET = "ai2/atec-olmoearth"
BEAKER_IMAGE = "favyen/forest_loss_driver"  # nosec
GCP_CREDENTIALS_SECRET = "RSLEARN_GCP_CREDENTIALS"  # nosec
GOOGLE_CLOUD_PROJECT = "earthsystem-dev-c3po"  # nosec
WEKA_BUCKET_NAME = "dfive-default"


if __name__ == "__main__":
    with Beaker.from_env(default_workspace=WORKSPACE) as beaker:
        task_name = "forest_loss_driver_inference_" + str(uuid.uuid4())[0:8]
        logger.info("Selected task name %s", task_name)

        # Various env vars for GCP access.
        gcp_credentials_mount_path = "/etc/credentials/gcp_credentials.json"
        env_vars = [
            BeakerEnvVar(
                name="GOOGLE_APPLICATION_CREDENTIALS",  # nosec
                value=gcp_credentials_mount_path,  # nosec
            ),
            BeakerEnvVar(
                name="GCLOUD_PROJECT",  # nosec
                value=GOOGLE_CLOUD_PROJECT,  # nosec
            ),
            BeakerEnvVar(
                name="GOOGLE_CLOUD_PROJECT",  # nosec
                value=GOOGLE_CLOUD_PROJECT,  # nosec
            ),
            BeakerEnvVar(
                name="STUDIO_API_KEY",  # nosec
                secret="STUDIO_API_KEY",  # nosec
            ),
            # olmoearth_datasets API, used to identify Sentinel-2 assets for
            # visualization (instead of the rate-limited Planetary Computer STAC API).
            BeakerEnvVar(
                name="OEDATASETS_API_URL",  # nosec
                value="https://datasets.olmoearth.allenai.org",  # nosec
            ),
            BeakerEnvVar(
                name="DATASETS_API_TOKEN",  # nosec
                secret="LCC_DATASETS_API_TOKEN",  # nosec
            ),
        ]
        datasets = [
            BeakerDataMount(
                source=BeakerDataSource(secret=GCP_CREDENTIALS_SECRET),  # nosec
                mount_path=gcp_credentials_mount_path,  # nosec
            ),
            BeakerDataMount(
                source=BeakerDataSource(weka=WEKA_BUCKET_NAME),
                mount_path=f"/weka/{WEKA_BUCKET_NAME}",
            ),
        ]

        command = [
            "python",
            "-m",
            "olmoearth_projects.main",
            "projects.forest_loss_driver.deploy",
            "integrated_pipeline",
            "--config",
            "olmoearth_run_data/forest_loss_driver/deploy.yaml",
        ]

        experiment_spec = BeakerExperimentSpec.new(
            budget=BEAKER_BUDGET,
            task_name=task_name,
            beaker_image=BEAKER_IMAGE,
            priority=BeakerJobPriority.normal,
            cluster="ai2/jupiter",
            command=command,
            env_vars=env_vars,
            datasets=datasets,
            # The job is preemptible, but protected from preemption for the first
            # 8 hours (the maximum allowed on most clusters). If it is preempted
            # after that, it is re-queued and the pipeline restarts from scratch.
            min_runtime="8h",
            auto_resume=True,
        )
        logger.info(f"Creating experiment: {task_name}")
        beaker.experiment.create(name=task_name, spec=experiment_spec)
