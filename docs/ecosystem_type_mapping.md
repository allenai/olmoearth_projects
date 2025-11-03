## Ecosystem Type Mapping

OlmoEarth-v1-FT-EcosystemTypeMapping-Base is a model fine-tuned from OlmoEarth-v1-Base
on expert-annotated ecosystem type data provided by [Global Ecosystem Atlas](https://globalecosystemsatlas.org/).
It is trained specifically for the north Africa region. The categories correspond to those in
the [IUCN Gloabl Ecosystem Typology](https://global-ecosystems.org/page/typology).

Here are relevant links for fine-tuning and applying the model per the documentation in
[the main README](../README.md):

- Model checkpoint: https://huggingface.co/allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base/blob/main/model.ckpt

## Model Details

The model inputs six timesteps of Sentinel-2 L2A satellite images, with one mosaic per
30-day period over a 270-day time range (some timesteps may be skipped if not enough
Sentinel-2 images are available).

It processes each 32x32 crop of the input image separately, and predicts the
predominant ecosystem type in each crop.

It achieves 64.8% accuracy on our test set.

## Training Data

The model is trained on ecosystem type data from [Global Ecosystem Atlas](https://globalecosystemsatlas.org/).
They will release the dataset in 2026.

## Inference

Inference is documented in [the main README](../README.md). The 180-day time range
starting at the start timestamp in the prediction request geometry will be used to
obtain the Sentinel-2 30-day mosaics; images from the preceding 90 days may be used if
there are some 30-day periods during the 180-day time range with no Sentinel-2
coverage. The end timestamp won't be used and can be set arbitrarily, e.g. set 180 days
after the start timestamp.
