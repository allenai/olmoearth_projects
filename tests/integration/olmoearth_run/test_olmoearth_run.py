"""Test olmoearth_run pipeline."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely
import yaml
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from olmoearth_projects.olmoearth_run.olmoearth_run import olmoearth_run


def test_olmoearth_run_solar_farm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test OlmoEarthRun pipeline by applying solar farm on small request geometry."""
    # For now this is fixed but we should figure out how to have standardized path for
    # each application later.
    checkpoint_path = "gs://ai2-rslearn-projects-data/projects/2025_06_06_helios_finetuning/v2_satlas_solar_farm_128_ts_helios_per_mod_patchdisc_contrastive_fix_esrun/checkpoints/epoch=9999-step=99999.ckpt"

    # Copy the configuration files. We use the tmp_path as the config dir that we will
    # initialize from since we will customize the request geometry.
    src_dir = Path("olmoearth_run_data/satlas_solar_farm/")
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)

    # Copy dataset.json without modifications.
    with (
        (src_dir / "dataset.json").open("rb") as src,
        (config_dir / "dataset.json").open("wb") as dst,
    ):
        shutil.copyfileobj(src, dst)

    # This request geometry should be at least 10% solar farm (and at most 50%).
    # It is centered at a solar farm and we extend beyond the solar farm.
    solar_farm_center = (-111.613, 33.267)
    request_geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(
            solar_farm_center[0] - 0.01,
            solar_farm_center[1] - 0.01,
            solar_farm_center[0] + 0.01,
            solar_farm_center[1] + 0.01,
        ),
        None,
    )
    feat = Feature(
        request_geometry,
        {
            "oe_start_time": datetime(2024, 12, 1, tzinfo=UTC).isoformat(),
            "oe_end_time": datetime(2025, 7, 1, tzinfo=UTC).isoformat(),
        },
    )
    GeojsonVectorFormat().encode_to_file(
        UPath(config_dir / "prediction_request_geometry.geojson"), [feat]
    )

    # We also customize the olmoearth_run.yaml since we want to use a single window.
    with (src_dir / "olmoearth_run.yaml").open() as f:
        olmoearth_run_config = yaml.safe_load(f)
    olmoearth_run_config["partition_strategies"]["partition_request_geometry"] = {
        "class_path": "olmoearth_run.runner.tools.partitioners.noop_partitioner.NoopPartitioner",
    }
    olmoearth_run_config["partition_strategies"]["prepare_window_geometries"] = {
        "class_path": "olmoearth_run.runner.tools.partitioners.reprojection_partitioner.ReprojectionPartitioner",
        "init_args": {
            "output_projection": {
                "class_path": "rslearn.utils.geometry.Projection",
                "init_args": {
                    "crs": "EPSG:3857",
                    "x_resolution": 10,
                    "y_resolution": -10,
                },
            },
            "use_utm": True,
        },
    }
    with (config_dir / "olmoearth_run.yaml").open("w") as f:
        yaml.safe_dump(olmoearth_run_config, f)

    # We customize the model.yaml to use smaller batch size.
    # Because in CI we have small system memory.
    with (src_dir / "model.yaml").open() as f:
        model_config = yaml.safe_load(f)
    model_config["data"]["init_args"]["batch_size"] = 1
    with (config_dir / "model.yaml").open("w") as f:
        yaml.safe_dump(model_config, f)

    # Disable W&B for this test.
    monkeypatch.setenv("WANDB_MODE", "disabled")
    # Set custom extra files directory that contains the requisite Helios config.
    monkeypatch.setenv(
        "EXTRA_FILES_PATH", "gs://ai2-rslearn-projects-data/helios/checkpoints/"
    )

    scratch_dir = tmp_path / "scratch"
    olmoearth_run(
        config_path=config_dir,
        scratch_path=scratch_dir,
        checkpoint_path=checkpoint_path,
    )

    result_fname = scratch_dir / "results" / "results_raster" / "result_0.tif"
    assert result_fname.exists()
    with rasterio.open(result_fname) as raster:
        array = raster.read()
    assert 0.1 < (np.count_nonzero(array == 1) / np.size(array)) < 0.5
