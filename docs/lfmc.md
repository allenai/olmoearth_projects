## Live Fuel Moisture Content (LFMC) Mapping

OlmoEarth-v1-FT-LFMC-Base is a model fine-tuned from OlmoEarth-v1-Base for predicting
the live fuel moisture content of woody vegetation from Sentinel-2 and Sentinel-1
satellite images.

Here are relevant links for fine-tuning and applying the model per the documentation in
[the main README](../README.md):

- Model checkpoint: https://huggingface.co/allenai/OlmoEarth-v1-FT-LFMC-Base/resolve/main/model.ckpt
- Annotation GeoJSONs: [[annotation_features.geojson](https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/lfmc/20251029/annotation_features.geojson) [annotation_task_features.geojson](https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/lfmc/20251029/annotation_task_features.geojson)]
- rslearn dataset: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/lfmc/20251029/dataset.tar

## Model Details

The model inputs twelve timesteps of satellite image data, with one
[Sentinel-1 RTC](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc)
mosaic and one
[Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)
mosaic per 14-day period.

At each pixel, it regresses the LFMC of woody vegetation.

It achieves a mean squared error of 580.5 on our validation set.

## Training Data

The model is trained on the Globe-LFMC 2.0 dataset. We use the subset of the data in
the continental US.

Each sample in the dataset specifies a longitude, latitude, timestamp, fuel type, and
LFMC value. We only use the woody fuel type subset for this model. For each sample, we
create an rslearn window centered at the sample's longitude/latitude and with time
range equal to the 168 days ending at the sample's timestamp. We use rslearn to obtain
twelve Sentinel-2 and Sentinel-1 images during that time range, with one per 14-day period.

We split the dataset into train, val, and test splits spatially, where 128x128 pixel
grid cells are assigned via hash to train (70%), val (20%), or test (10%).

## Inference

Inference is documented in [the main README](../README.md). The prediction request
geometry should have start timestamp equal to the timestamp for which you want to make
the LFMC prediction (e.g., the current timestamp). The end timestamp won't be used and
can be set arbitrarily, e.g. set equal to the start timestamp.

## Fine-tuning

Fine-tuning is documented in [the main README](../README.md).
