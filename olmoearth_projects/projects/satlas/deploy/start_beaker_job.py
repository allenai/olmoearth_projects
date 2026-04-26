"""Start Beaker jobs for Satlas prediction pipeline.

Launches one Beaker experiment per application (marine_infra, wind_turbine,
solar_farm). Typically executed from the GitHub Action:
.github/workflows/satlas_prediction.yaml
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
BEAKER_BUDGET = "ai2/es-platform"
BEAKER_IMAGE = "favyen/satlas_deploy"  # nosec
GCP_CREDENTIALS_SECRET = "RSLEARN_GCP_CREDENTIALS"  # nosec
GOOGLE_CLOUD_PROJECT = "earthsystem-dev-c3po"  # nosec
WEKA_BUCKET_NAME = "dfive-default"

APPLICATIONS = ["marine_infra", "wind_turbine", "solar_farm"]


if __name__ == "__main__":
    with Beaker.from_env(default_workspace=WORKSPACE) as beaker:
        for app in APPLICATIONS:
            task_name = f"satlas_{app}_" + str(uuid.uuid4())[0:8]
            logger.info("Starting Beaker job: %s", task_name)

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
                "projects.satlas.deploy",
                "integrated_pipeline",
                "--config",
                f"olmoearth_run_data/satlas/{app}/deploy.yaml",
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
                preemptible=False,
            )
            logger.info("Creating experiment: %s", task_name)
            beaker.experiment.create(name=task_name, spec=experiment_spec)
