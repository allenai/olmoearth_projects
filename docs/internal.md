## Ai2-Internal Documentation

## Model Development Workflow

This section covers where project-specific code, configs, and rslearn datasets should
be stored while a model is being developed.

For projects that need to use rslearn directly initially:
1. Add code like data curation and programmatic window creation to `rslp/[project_name]`
   in rslearn_projects.
2. Add rslearn dataset and model configs to `data/[project_name]/[version_id]/` (in
   rslearn_projects). Document the version history in data/[project_name]/README.md.
3. Put the rslearn dataset in `/weka/dfive-default/rslearn-eai/datasets/[project_name]/[version_id]`.
   Document the available rslearn datasets (version history) in `data/[project_name]/README.md`.
4. Run data materialization and fine-tuning jobs from rslearn_projects. See
   `rslp/common/README.md` for some details about how to launch these jobs.

Once ready for fine-tuning and/or prediction runs in OlmoEarth platform:
1. Copy configs to `olmoearth_run_data/[project_name]/[version_id]` in
   olmoearth_projects. `olmoearth_run_data/[project_name]/` should document where the
   configs came from and anything special about the olmoearth_run.yaml.
2. Add any new supporting code relevant for deploying on OlmoEarth platform to
   `olmoearth_projects/[project_name]`.
3. If running inference only, copy the checkpoint from WEKA to
   `gs://rslearn-eai/model_checkpoints/[project_name]/[version].ckpt` so that the
   platform has access to it.

Also see `docs/internal.md` in rslearn_projects for basic info about using that
repository.

## WEKA Dataset Locations

### LFMC

- Dataset used for training: `/weka/dfive-default/rslearn-eai/datasets/lfmc/20251023/woody/scratch/dataset`
- Here is a copy that should be same as above but maybe with some unneeded layers removed: `/weka/dfive-default/olmoearth_release_data/rslearn_datasets/lfmc/`

Note that olmoearth_evals uses `/weka/dfive-default/rslearn-eai/datasets/lfmc/20250626/`
which is an older version of the dataset.

### Ecosystem Type Mapping

- Dataset used for training: `/weka/dfive-default/rslearn-eai/datasets/geo/dataset_v2/dataset/`

### Forest Loss Driver

- Dataset used for training: `/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined/`

### Mangrove Classification

- Dataset used for training: `/weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250626/`

### Solar Farm Segmentation

- Dataset used for training: `/weka/dfive-default/rslearn-eai/datasets/solar_farm/dataset_v1/20250605/`
