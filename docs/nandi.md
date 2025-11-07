## Nandi Crop Type Mapping

OlmoEarth-v1-FT-Nandi-Base is a model fine-tuned from OlmoEarth-v1-Base for predicting crop and land-cover type across the Nandi county in Kenya using Sentinel-2 satellite images.

Here are relevant links for fine-tuning and applying the model per the documentation in
[the main README](../README.md):

- Model checkpoint: https://huggingface.co/allenai/OlmoEarth-v1-FT-Nandi-Base/blob/main/model.ckpt
- Annotation GeoJSONs: https://huggingface.co/datasets/allenai/olmoearth_projects_nandi/tree/main
- rslearn dataset: https://huggingface.co/datasets/allenai/olmoearth_projects_nandi/blob/main/dataset.tar

## Model Details

The model inputs twelve timesteps of satellite image data, one [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) mosaic per 30-day period.

The model is trained to predict crop and land-cover type for every pixel within each 16×16 input patches.

The model (window size: 16x16, patch size: 1) achieves an overall accuracy of 87.3% on our validation set. The table below summarizes our experiments with different patch sizes and input modalities. Overall, mnodels using patch size 1 perform the best.

| Window Size | Patch Size | Modalities | Accuracy (%) |
|--------------|-------------|-------------|---------------|
| 16×16 | 1 | Sentinel-2 | 87.3 |
| 16×16 | 1 | Sentinel-2 + Sentinel-1 | TBD |
| 16×16 | 2 | Sentinel-2 | 86.5 |
| 16×16 | 2 | Sentinel-2 + Sentinel-1 | 86.7 |
| 16×16 | 4 | Sentinel-2 | 81.9 |
| 16×16 | 4 | Sentinel-2 + Sentinel-1 | 82.2 |

## Training Data

The model is trained on ground-truth labels collected by [CGIAR/IFPRI](https://www.ifpri.org/). The original dataset includes 819 labeled polygons, from which we sampled training points. To improve coverage, we added extra point samples from ESA WorldCover (since the original dataset lacked Water and Built-up classes) and additional Tree samples annotated in the Studio to correct misclassification of natural forest areas as Coffee.

In total, the dataset covers 10 categories: coffee, maize, sugarcane, tea, vegetables, legumes, grassland, trees, water, and built-up.

Each sample includes its longitude, latitude, time range (2022-09 to 2023-09), and crop or land-cover type. For each sample, we generate an rslearn window centered on the location, covering one year of data. We use rslearn to obtain twelve Sentinel-2 and Sentinel-1 imagery during that time range, with one per 30-day period.

The dataset is split spatially into training (75%) and validation (25%) sets, based on a 128×128-pixel grid hashed into the two splits.

## Inference

Inference is documented in [the main README](../README.md). The prediction request geometry should have start and end timestamps that covers one year, ideally from 2022-09-01 to 2023-09-01 to match the training data. However, you can also run inference for other one-year periods, such as 2018-09-01 to 2019-09-01. Inference runs on all 1024×1024 grid cells intersecting the geometry, using satellite images from the specified time range.

Here's the [inference output](https://olmoearth.allenai.org/viewer/6b1e4537-ea68-47f3-9a11-61ca2d468fd0#9.55/0.2218/35.1037) for the whole Nandi county.

## Fine-tuning

Fine-tuning is documented in [the main README](../README.md).
