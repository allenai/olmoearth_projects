## Forest Loss Driver Classification

OlmoEarth-v1-FT-ForestLossDriver-Base is a model fine-tuned from OlmoEarth-v1-Base for
classifying forest loss drivers. It is trained to operate over
[GLAD-S2 forest loss alerts](https://data.globalforestwatch.org/datasets/gfw::integrated-deforestation-alerts/about),
which are updated weekly and report the locations of forest loss. Thus, instead of
detecting forest loss from scratch, we take connected components of GLAD-S2 forest loss
pixels and extend them with a driver classification that predicts the cause of the
forest loss.

The driver categories are:

- Agriculture
- Mining
- Airstrip
- Road
- Logging
- Burned
- Landslide
- Hurricane
- River
- None

Here are relevant links for fine-tuning and applying the model per the documentation in
[the main README](../README.md):

- Model checkpoint: https://huggingface.co/allenai/OlmoEarth-v1-FT-ForestLossDriver-Base/resolve/main/model.ckpt
- rslearn dataset: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/forest_loss_driver/20251029/dataset.tar

## Model Details

For each connected component of GLAD-S2 forest loss pixels, the model inputs two image
time series that are 64x64 pixels (at 10 m/pixel) centered at the center of the
connected component. The first time series consists of four Sentinel-2 L2A images
captured before the forest loss, while the second time series consists of four
Sentinel-2 L2A images captured after the forest loss.

The model classifies the forest loss driver, with 10 classes (see above). It achieves
an accuracy of 76.1% on our validation set. Here is the confusion matrix:

| Category  | Ag | Airstrip | Burned | Hurricane | Landslide | Logging | Mining | None | River | Road |
| --------  | -- | -------- | ------ | --------- | --------- | ------- | ------ | ---- | ----- | ---- |
| Ag        | 37 |      0   |      2 |         0 |         0 |       0 |      0 |    2 |     0 |    3 |
| Airstrip  |  0 |      0   |      0 |         0 |         0 |       0 |      0 |    0 |     0 |    0 |
| Burned    |  2 |      0   |     21 |         3 |         0 |       0 |      0 |    4 |     0 |    0 |
| Hurricane |  0 |      0   |      0 |         6 |         0 |       0 |      0 |    0 |     0 |    0 |
| Landslide |  0 |      0   |      0 |         0 |         1 |       0 |      0 |    0 |     0 |    0 |
| Logging   |  0 |      0   |      0 |         2 |         0 |       2 |      0 |    0 |     0 |    0 |
| Minning   |  0 |      0   |      0 |         0 |         0 |       0 |      1 |    0 |     0 |    0 |
| None      |  2 |      0   |      1 |         1 |         0 |       2 |      0 |   11 |     0 |    1 |
| River     |  0 |      0   |      0 |         0 |         0 |       0 |      0 |    1 |     0 |    0 |
| Road      |  0 |      0   |      0 |         0 |         0 |       0 |      0 |    0 |     0 |    4 |

## Training Data

The model is trained on forest loss driver annotations produced by Amazon Conservation
Association. Each annotation specifies a polygon and timestamp that originate from a
GLAD-S2 alert, along with the driver category. We use rslearn to obtain the four
pre-forest-loss Sentinel-2 L2A images and four post-forest-loss images.

We split the dataset into 75% train and 25% val.

## Inference

Inference is documented in [the main README](../README.md). The prediction request
geometry consists of one GeoJSON polygon for each connected component of GLAD-S2 alert
pixels that should be processed.

The prediction request geometry can be generated from the
[GLAD alert files on GCS](https://console.cloud.google.com/storage/browser/earthenginepartners-hansen/S2alert/):

```
python -m olmoearth_projects.main projects.forest_loss_driver extract_alerts --extract_alerts_args.gcs_tiff_filenames+=080W_20S_070W_10S.tif --extract_alerts_args.out_fname='prediction_request_geometry.geojson' --extract_alerts_args.days=90
```

Here, the `gcs_tiff_filenames` is a list of GLAD-S2 tiles to process (see the GCS link
above for the available tiles) and `days` specifies the time range (it will cover this
many days from the current timestamp into the past).

If you open the `prediction_request_geometry.geojson` in qgis, you should see several
small polygons. The model will be applied on a 128x128 pixel window centered at each of
these polygons.

To run inference:

```
mv prediction_request_geometry.geojson olmoearth_run_data/forest_loss_driver/prediction_request_geometry.geojson
mkdir -p ./checkpoints
wget -O checkpoints/forest_loss_driver.ckpt https://huggingface.co/allenai/OlmoEarth-v1-FT-ForestLossDriver-Base/blob/main/model.ckpt
export NUM_WORKERS=32
export WANDB_PROJECT=forest_loss_driver
export WANDB_NAME=forest_loss_driver
export WANDB_ENTITY=YOUR_WANDB_ENTITY
python -m olmoearth_projects.main olmoearth_run olmoearth_run --config_path $PWD/olmoearth_run_data/forest_loss_driver/ --checkpoint_path $PWD/checkpoints/forest_loss_driver.ckpt --scratch_path project_data/forest_loss_driver/
```

## Fine-tuning

Fine-tuning is documented in [the main README](../README.md).
