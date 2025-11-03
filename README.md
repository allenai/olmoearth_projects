## OlmoEarth Projects

This repository contains configuration files, model checkpoint references, and
documentation for several remote sensing models built on top of OlmoEarth at Ai2. It
also includes tooling and tutorials for building new models using various components of
OlmoEarth.

The models available here are:

- [Live Fuel Moisture Content Mapping](docs/lfmc.md)
- [Forest Loss Driver Classification](docs/forest_loss_driver.md)
- [Mangrove Mapping](docs/mangrove.md)
- [Ecosystem Type Mapping](docs/ecosystem_type_mapping.md)
- [Land Use / Land Cover Mapping in Southern Kenya](docs/awf.md)

The links above provide more details about the training data and intended use case for
each model.

Here are tutorials for applying OlmoEarth for new tasks:

- [Fine-tuning OlmoEarth for Segmentation](docs/tutorials/FinetuneOlmoEarthSegmentation.md)
- [Computing Embeddings using OlmoEarth](https://github.com/allenai/rslearn/blob/master/docs/examples/OlmoEarthEmbeddings.md)
- [Fine-tuning OlmoEarth in rslearn](https://github.com/allenai/rslearn/blob/master/docs/examples/FinetuneOlmoEarth.md)

These tutorials use all or a subset of the components of OlmoEarth:

- [olmoearth_pretrain](https://github.com/allenai/olmoearth_pretrain/), the OlmoEarth
  pre-trained model.
- [rslearn](https://github.com/alleani/rslearn/), our tool for obtaining satellite
  images and other geospatial data from online data sources, and for fine-tuning
  remote sensing foundation models.
- [olmoearth_run](https://pypi.org/project/olmoearth-runner/), our higher-level
  infrastructure that automates various steps on top of rslearn such as window creation
  and inference post-processing.

## Installation

We recommend installing using uv. See
[Installing uv](https://docs.astral.sh/uv/getting-started/installation/) for
instructions to install uv. Once uv is installed:

```
git clone https://github.com/allenai/olmoearth_projects.git
cd olmoearth_projects
uv sync
source .venv/bin/activate
```

## Applying Existing Models

There are three steps to applying the models in this repository:

1. Customize the prediction request geometry, which specifies the spatial and temporal
   extent to run the model on.
2. Execute the olmoearth_run steps to build an rslearn dataset for inference, and to
   apply the model on the dataset.
3. Collect and visualize the outputs.

### Customizing the Prediction Request Geometry

The configuration files for each project are stored under
`olmoearth_run_data/PROJECT_NAME/`. There are three configuration files:

- `dataset.json`: this is an rslearn dataset configuration file that specifies the
  types of satellite images that need to be downloaded to run the model, and how to
  obtain them. Most models rely on some combination of Sentinel-1 and Sentinel-2
  satellite images, and are configured to download those images from Microsoft
  Planetary Computer.
- `model.yaml`: this is an rslearn model configuration file that specifies the model
  architecture, fine-tuning hyperparameters, data loading steps, etc.
- `olmoearth_run.yaml`: this is an olmoearth_run configuration file that specifies how
  the prediction request geometry should be translated into rslearn windows, and how
  the inference outputs should be combined together.

Some projects also include an example `prediction_request_geometry.geojson`, but this
will need to be modified to specify your target region. The spatial extent is specified
with standard GeoJSON features; you can use [geojson.io](https://geojson.io/) to draw
polygons on a map and get the corresponding GeoJSON. The temporal extent is specified
using properties on each feature:

```jsonc
{
  "type": "FeatureCollection",
  "properties": {},
  "features": [
    {
      "type": "Feature",
      "geometry": {
        // ...
      },
      "properties": {
        "oe_start_time": "2024-01-01T00:00:00+00:00",
        "oe_end_time": "2024-02-01T00:00:00+00:00"
      },
    }
  ]
}
```

Here, the `oe_start_time` and `oe_end_time` indicate that the prediction for the
location of this feature should be based on satellite images around January 2024. The
per-model documentation details how these timestamps should be chosen. Some models like
forest loss driver classification provide project-specific tooling for generating the
prediction request geometry.

### Executing olmoearth_run

Consult the per-model documentation to download the associated fine-tuned model
checkpoint. For example:

```
mkdir ./checkpoints
wget https://huggingface.co/allenai/OlmoEarth-v1-FT-LFMC-Base/resolve/main/model.ckpt -O checkpoints/lfmc.ckpt
```

Set needed environment variables:

```
export NUM_WORKERS=32
export WANDB_PROJECT=lfmc
export WANDB_NAME=lfmc_inference_run
export WANDB_ENTITY=YOUR_WANDB_ENTITY
```

Then, execute olmoearth_run:

```
mkdir ./project_data
python -m olmoearth_projects.main olmoearth_run olmoearth_run --config_path $PWD/olmoearth_run_data/lfmc/ --checkpoint_path $PWD/checkpoints/lfmc.ckpt --scratch_path project_data/lfmc/
```

### Visualizing Outputs

The results directory (`project_data/lfmc/results/results_raster/` in the example)
should be populated with one or more GeoTIFFs. You can visualize this in GIS software
like qgis:

```
qgis project_data/lfmc/results/results_raster/*.tif
```

## Reproducing Fine-tuning for Existing Models

We have released model checkpoints for each of the fine-tuned models in this
repository, but you can reproduce the model by fine-tuning the pre-trained OlmoEarth
checkpoint on each task training dataset.

First, consult the per-model documentation above for the URL of the rslearn dataset tar
file, and download and extract it. For example, for the LFMC model:

```
wget https://huggingface.co/datasets/allenai/olmoearth_projects_lfmc/blob/main/dataset.tar
tar xvf dataset.tar
```

Set environment variables expected by the fine-tuning procedure (uses W&B)

```
export DATASET_PATH=/path/to/extracted/data/
export NUM_WORKERS=32
export TRAINER_DATA_PATH=./trainer_data
export PREDICTION_OUTPUT_LAYER=output
export WANDB_PROJECT=olmoearth_projects
export WANDB_NAME=my_training_run
export WANDB_ENTITY=...
```

Then run fine-tuning using the model configuration file in the `olmoearth_run_data`,
e.g.:

```
rslearn model fit --config olmoearth_run_data/lfmc/model.yaml
```

Losses and metrics should then be logged to your W&B. The checkpoint would be saved in
the TRAINER_DATA_PATH (e.g. `./trainer_data`); two checkpoints should be saved, the
latest checkpoint (`last.ckpt`) and the best checkpoint (`epoch=....ckpt`). You can use
the best checkpoint for the Applying Existing Models section in lieu of the checkpoint
that we proivde.

If training fails halfway, you can resume it from `last.ckpt`:

```
rslearn model fit --config olmoearth_run_data/lfmc/model.yaml --ckpt_path $TRAINER_DATA_PATH/last.ckpt
```
