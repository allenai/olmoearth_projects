## Oil Spill Detection

OlmoEarth-v1-FT-OilSpills-Base is a model fine-tuned from OlmoEarth-v1-Base for
detecting oil slicks in ocean waters from Sentinel-1 satellite images.

Here are relevant links for fine-tuning and applying the model per the documentation in
[the main README](../README.md):

- Model checkpoint: TBD
- Annotation GeoJSONs: [[annotation_features.geojson](https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/oil_spills/annotation_features.geojson)] [[annotation_task_features.geojson](https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/oil_spills/annotation_task_features.geojson)]
- Labels: [labels.tar.gz](https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/oil_spills/labels.tar.gz)
- rslearn dataset: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/oil_spills/dataset.tar.gz

## Model Details

The model inputs a single timestep of Sentinel-1 RTC data (VV and VH polarizations)
at 10m resolution, fetched via the
[OlmoEarth dataset service](https://datasets-staging.olmoearth.allenai.org).

At each pixel, it predicts a binary segmentation mask: slick (oil present) or not_slick
(no oil). The model uses 128x128 pixel windows.

Architecture:
- Encoder: OlmoEarth-v1-Base (patch size 4, embedding size 768)
- Decoder: UNet with 2 output classes
- Transforms: Sentinel1ToDecibels, OlmoEarthNormalize

## Training Data

The positive labels come from [SkyTruth](https://skytruth.org/), which provides
multipolygon annotations of oil slicks derived from Sentinel-1 imagery. Each annotation
includes the slick geometry and the timestamp of the Sentinel-1 acquisition it was
identified in. The labels span from 2016 to 2026.

Negative samples are generated on defined rings outside of each true positive slick
using `create_oil_spill_dataset.py`. This ensures the model sees examples of ocean
surface without oil near known spill locations.

Because the labels are derived from Sentinel-1 data, the model is trained exclusively on
Sentinel-1 imagery to match the known acquisition timestamps. Sentinel-2 overlap and
other data sources can be added later for additional coverage.

### Dataset Statistics

- 3,786 positive examples (oil slicks)
- 3,786 negative examples (generated)
- 7,572 total labeled examples

After windowing at 128x128 pixels with 10m resolution and spatial splitting:

| Split | Windows |
|-------|---------|
| Train | ~166,180 |
| Val   | ~36,724 |
| Test  | ~36,884 |

The dataset is split spatially using a 10km grid (70% train / 15% val / 15% test) to
avoid data leakage between geographically nearby samples.

### Data Processing Pipeline

1. **Label preparation**: Clean SkyTruth labels and generate negative samples
   (`olmoearth_projects/projects/oil_spills/create_oil_spill_dataset.py`)
2. **Annotation creation**: Convert labels to olmoearth_run format
   (`scripts/oer_annotation_creation.py`)
3. **Window preparation**: Create rslearn windows with spatial splits
   (`prepare_labeled_windows`)
4. **Dataset materialization**: Download Sentinel-1 imagery via the OlmoEarth dataset
   service for each window (`build_dataset_from_windows`)

## Reproducing

### Environment Setup

```bash
cd ~/olmoearth_projects
source .venv/bin/activate
export PROJECT_PATH=~/olmoearth_projects/olmoearth_run_data/oil_spills
export OER_DATASET_PATH=~/olmoearth_projects/oerun_dataset/oil_spills
export GROUP_NAME="spatial_split_10km"
export OEDATASETS_API_URL=https://datasets-staging.olmoearth.allenai.org
```

### Step 1: Prepare Labels

Generate the positive/negative dataset from SkyTruth labels:

```bash
python olmoearth_projects/projects/oil_spills/create_oil_spill_dataset.py
```

The cleaned labels are also available in GCS as `labels.tar.gz`.

### Step 2: Create Annotations

Convert the labels GeoJSON into olmoearth_run annotation format:

```bash
python scripts/oer_annotation_creation.py \
  --input $PROJECT_PATH/labels.geojson \
  --output-dir $PROJECT_PATH
```

### Step 3: Prepare Windows

```bash
python -m olmoearth_projects.main olmoearth_run prepare_labeled_windows \
  --project_path $PROJECT_PATH \
  --scratch_path $OER_DATASET_PATH
```

Verify the splits:

```bash
grep -roh '"split": "[^"]*"' $OER_DATASET_PATH/dataset/windows/$GROUP_NAME/ \
  --include="metadata.json" | sort | uniq -c
```

### Step 4: Build Dataset

Download Sentinel-1 imagery via the OlmoEarth dataset service for each window:

```bash
python -m olmoearth_projects.main olmoearth_run build_dataset_from_windows \
  --project_path $PROJECT_PATH \
  --scratch_path $OER_DATASET_PATH
```

### Step 5: Fine-tune

Set wandb environment variables:

```bash
export WANDB_PROJECT=oe_oil_spill_detection
export WANDB_NAME=slick_segmentation_s1
export WANDB_ENTITY=your-wandb-entity
```

Run fine-tuning:

```bash
python -m olmoearth_projects.main olmoearth_run finetune \
  --project_path $PROJECT_PATH \
  --scratch_path $OER_DATASET_PATH
```

### Step 6: Run Predictions on Validation Set

After training, run predictions on the holdout validation set using
`model_predict_validation.yaml`. This writes predictions to the `output` layer in
each validation window:

```bash
export DATASET_PATH=$OER_DATASET_PATH/dataset
export CHECKPOINT_PATH=$OER_DATASET_PATH/checkpoints/best.ckpt

rslearn model predict \
  --config $PROJECT_PATH/model_predict_validation.yaml \
  --ckpt_path $CHECKPOINT_PATH
```

### Step 7: Calculate Metrics

Run the confusion matrix and metrics script against the validation predictions:

```bash
python olmoearth_projects/projects/oil_spills/calculate_metrics.py \
  $OER_DATASET_PATH/dataset/windows/$GROUP_NAME
```

This outputs pixel-level and window-level metrics including accuracy, precision,
recall, F1-score, IoU, and a confusion matrix. It also saves example windows for
each category (TP, FP, FN, TN) for visual inspection.

## Inference

Inference is documented in [the main README](../README.md). The prediction request
geometry should specify the area of interest with start and end timestamps matching a
Sentinel-1 acquisition time. The model uses a 6-hour duration window with a 3-hour
time offset to find matching Sentinel-1 scenes via the OlmoEarth dataset service.

Output classification:
- 0: not_slick (red)
- 1: slick (blue)

## Fine-tuning

Fine-tuning is documented in [the main README](../README.md).
