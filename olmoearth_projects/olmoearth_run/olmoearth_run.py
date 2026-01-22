"""Run OlmoEarthRunPredictRunner inference pipeline."""

import hashlib
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path

import fsspec
from olmoearth_run.runner.local.fine_tune_runner import OlmoEarthRunFineTuneRunner
from olmoearth_run.runner.local.predict_runner import OlmoEarthRunPredictRunner
from upath import UPath

from olmoearth_projects.utils.logging import get_logger

logger = get_logger(__name__)


def get_local_checkpoint(checkpoint_path: UPath) -> Path:
    """Get a local path to the specified checkpoint file, caching it locally if needed.

    Args:
        checkpoint_path: a UPath to the checkpoint file.

    Returns:
        a local UPath, which is the same as checkpoint_path if it is already local, or
            points to a cached version in the system temporary directory.
    """
    # Cache the checkpoint if it isn't already local.
    if isinstance(checkpoint_path.fs, fsspec.implementations.local.LocalFileSystem):
        logger.info("using local checkpoint at %s", checkpoint_path)
        return Path(checkpoint_path)

    cache_id = hashlib.sha256(str(checkpoint_path).encode()).hexdigest()
    local_upath = (
        UPath(tempfile.gettempdir())
        / "rslearn_cache"
        / "olmoearth_run_checkpoints"
        / f"{cache_id}.ckpt"
    )

    if not local_upath.exists():
        logger.info("caching checkpoint from %s to %s", checkpoint_path, local_upath)
        local_upath.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("rb") as src, local_upath.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    logger.info("using cached checkpoint at %s", local_upath)
    return Path(local_upath)


def prepare_labeled_windows(project_path: Path, scratch_path: Path) -> None:
    """Run OlmoEarthRunFineTuneRunner's prepare_windows pipeline."""
    logger.info("Loading OlmoEarthRunFineTuneRunner")
    runner = OlmoEarthRunFineTuneRunner(
        project_path=project_path,
        scratch_path=scratch_path,
    )
    logger.info("Running prepare_labeled_windows")
    runner.prepare_labeled_windows()


def build_dataset_from_windows(project_path: Path, scratch_path: Path) -> None:
    """Run OlmoEarthRunFineTuneRunner's build_dataset_from_windows pipeline."""
    logger.info("Loading OlmoEarthRunFineTuneRunner")
    runner = OlmoEarthRunFineTuneRunner(
        project_path=project_path,
        scratch_path=scratch_path,
    )
    logger.info("Running build_dataset_from_windows")
    runner.build_dataset_from_windows()


def finetune(project_path: Path, scratch_path: Path) -> None:
    """Run EsFineTuneRunner finetune pipeline."""
    logger.info("Loading OlmoEarthRunFineTuneRunner")
    runner = OlmoEarthRunFineTuneRunner(
        project_path=project_path,
        scratch_path=scratch_path,
    )
    logger.info("Running finetune")
    runner.fine_tune()


def olmoearth_run(config_path: Path, scratch_path: Path, checkpoint_path: str) -> None:
    """Run EsPredictRunner inference pipeline.

    Args:
        config_path: directory containing the model.yaml, partition_strategies.yaml,
            and postprocessing_strategies.yaml configuration files.
        scratch_path: directory to use for scratch space.
        checkpoint_path: path to the model checkpoint.
    """
    runner = OlmoEarthRunPredictRunner(
        # OlmoEarth Run does not work with relative path, so make sure to convert to absolute here.
        project_path=config_path.absolute(),
        scratch_path=scratch_path,
        checkpoint_path=get_local_checkpoint(UPath(checkpoint_path)),
    )
    partitions = runner.partition()
    logger.info(f"Got {len(partitions)} partitions")

    logger.info("Building dataset across partitions")
    runner.build_dataset(partitions)

    for partition_id in partitions:
        logger.info(f"Running inference for partition {partition_id}")
        runner.run_inference(partition_id)
        logger.info(f"Postprocessing for partition {partition_id}")
        runner.postprocess(partition_id)

    logger.info("Combining across partitions")
    runner.combine(partitions)


class OlmoEarthRunStage(StrEnum):
    """The stage of olmoearth_run pipeline to run.

    We always run the partition stage so that is not an option here.
    """

    BUILD_DATASET = "build_dataset"
    RUN_INFERENCE = "run_inference"
    POSTPROCESS = "postprocess"
    COMBINE = "combine"


def one_stage(
    config_path: Path,
    scratch_path: Path,
    checkpoint_path: str,
    stage: OlmoEarthRunStage,
    partition_id: str | None = None,
) -> None:
    """Run OlmoEarthRunPredictRunner inference pipeline.

    Args:
        config_path: see olmoearth_run.
        scratch_path: see olmoearth_run.
        checkpoint_path: see olmoearth_run.
        stage: which stage to run.
        partition_id: the partition to run the stage for. If not set, we run the stage
            for all partitions, except BUILD_DATASET and COMBINE, which happens across
            partitions.
    """
    if stage == OlmoEarthRunStage.COMBINE and partition_id is not None:
        raise ValueError("partition_id cannot be set for COMBINE stage")

    runner = OlmoEarthRunPredictRunner(
        # OlmoEarth Run does not work with relative path, so make sure to convert to absolute here.
        project_path=config_path.absolute(),
        scratch_path=scratch_path,
        checkpoint_path=get_local_checkpoint(UPath(checkpoint_path)),
    )
    partitions = runner.partition()

    if stage == OlmoEarthRunStage.BUILD_DATASET:
        runner.build_dataset(partitions)

    if stage in [
        OlmoEarthRunStage.RUN_INFERENCE,
        OlmoEarthRunStage.POSTPROCESS,
    ]:
        fn = None
        if stage == OlmoEarthRunStage.RUN_INFERENCE:
            fn = runner.run_inference
        elif stage == OlmoEarthRunStage.POSTPROCESS:
            fn = runner.postprocess
        else:
            assert False

        if partition_id is not None:
            if partition_id not in partitions:
                raise ValueError(f"partition {partition_id} does not exist")
            fn(partition_id)
        else:
            for partition_id in partitions:
                fn(partition_id)

    elif stage == OlmoEarthRunStage.COMBINE:
        runner.combine(partitions)
