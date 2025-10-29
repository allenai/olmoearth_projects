## Mangrove Extent Mapping

OlmoEarth-v1-FT-Mangrove-Base is a model fine-tuned from OlmoEarth-v1-Base for preddicting mangrove extent from Sentinel-2.

Here are relevant links for fine-tuning and applying the model per the documentation in
[the main README](../README.md):

- Model checkpoint: https://huggingface.co/allenai/OlmoEarth-v1-FT-Mangrove-Base/resolve/main/model.ckpt
- Annotation GeoJsons: https://huggingface.co/allenai/olmoearth_projects_mangrove/blob/main/annotation_features.geojson
- rslearn dataset: https://huggingface.co/allenai/olmoearth_projects_mangrove/resolve/main/mangrove.tar

## Model Details

The model inputs twelve timesteps of satellite image data with one
mosaic [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)
mosaic per 30-day period.

At every 2 by 2 patch it outputs a classification of mangrove, water or other.

The model achieves strong performance on the validation set with an overall accuracy of 97.6%.
Mangrove classification achieves the F1 score of 98.7% (precision: 98.5%, recall: 99.0%),
followed by Water with an F1 score of 97.1% (precision: 96.5%, recall: 97.7%),
and Other with an F1 score of 96.3% (precision: 97.1%, recall: 95.4%).


## Training Data

The model is trained on data provided by Global Mangrove Watch available at https://zenodo.org/records/17394267 .


Each sample in the dataset specifies a longitude, latitude, a start and end time (1 year apart), and a class label. For each sample we create a 12 month time series of Sentinel 2 data within the time bounds.

We split the dataset into train, val, and test splits spatially, where each 1024 by 1024  pixel
grid cells are assigned via hash to train (50%), val (25%), or test (25%).

## Inference

Inference is documented in [the main README](../README.md). The prediction request
geometry should have start timestamp set 12 months prior to the date in which you would like to classify mangrove extent.
