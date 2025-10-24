## OlmoEarth Projects

This repository contains configuration files, model checkpoint references, and
documentation for several remote sensing models built on top of OlmoEarth at Ai2. It
also includes tooling and tutorials for building new models using various components of
OlmoEarth.

The models available here are:

- [Live Fuel Moisture Content Mapping](docs/lfmc.md)
- [Forest Loss Driver Classification](docs/forest_loss_drivers.md)
- [Mangrove Mapping](docs/mangrove.md)

TODO: models we are not sure if we can release:

- [Crop Type Mapping in Nandi County, Kenya](docs/nandi.md)
- [Land Use / Land Cover Mapping in Southern Kenya](docs/awf.md)

The links above provide more details about the training data and intended use case for
each model.

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

## Developing New Models

Here are tutorials for developing models using all of the components of OlmoEarth: the
[OlmoEarth model](https://github.com/allenai/olmoearth_pretrain/),
[rslearn](https://github.com/alleani/rslearn/), and
[olmoearth_run](https://github.com/allenai/olmoearth_run/).

- [Fine-tuning OlmoEarth for Classification](docs/tutorials/FinetuneOlmoEarthClassification.md)

We also have the following lighter-weight tutorials that use a subset of the OlmoEarth
components:

- For extracting embeddings from OlmoEarth, see the
  [OlmoEarth embedding guide](https://github.com/allenai/rslearn/blob/master/docs/examples/OlmoEarthEmbeddings.md)
  in rslearn.
- For fine-tuning OlmoEarth using rslearn, see the rslearn
  [OlmoEarth fine-tuning guide](https://github.com/allenai/rslearn/blob/master/docs/examples/FinetuneOlmoEarth.md).

## Applying Existing Models

There are three steps to applying the models in this repository:

1. Customize the prediction request geometry, which specifies the spatial and temporal
   extent to run the model on.
2. Execute the olmoearth_run steps to build an rslearn dataset for inference, and to
   apply the model on the dataset.
3. Collect and visualize the outputs.

### Customizing the Prediction Request Geometry

The configuration files for each project are stored under
`olmoearth_run_data/PROJECT_NAME/`. There are four configuration files:

- `prediction_request_geometry.geojson`: this is a GeoJSON that specifies the spatial
  and temporal extent for inference.
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

The `prediction_request_geometry.geojson` is an example and should be modified to
specify your target region. The spatial extent is specified with standard GeoJSON
features; you can use [geojson.io](https://geojson.io/) to draw polygons on a map and
get the corresponding GeoJSON. The temporal extent is specified using properties on
each feature:

```jsonc
{
  "type": "FeatureCollection"
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
wget https://huggingface.co/allenai/OlmoEarth-v1-Base-FT-LFMC/resolve/main/weights.pth -O checkpoints/lfmc.pth
```

Set needed environment variables:

```
export NUM_WORKERS=32
export WANDB_PROJECT=lfmc
export WANDB_NAME=YOUR_WANDB_ENTITY
```

Then, execute olmoearth_run:

```
mkdir ./project_data
python -m olmoearth_projects.main olmoearth_run olmoearth_run --config_path $PWD/olmoearth_run_data/lfmc/ --checkpoint_path $PWD/checkpoints/lfmc.pth --scratch_path project_data/lfmc/
```

### Visualizing Outputs

The results directory (`project_data/lfmc/results/results_raster/` in the example)
should be populated with one or more GeoTIFFs. You can visualize this in GIS software
like qgis:

```
qgis project_data/lfmc/results/results_raster/*.tif
```

## Fine-tuning

We have released model checkpoints for each of the fine-tuned models in this
repository, but you can reproduce the model by fine-tuning the pre-trained OlmoEarth
checkpoint on each task training dataset.

TODO
