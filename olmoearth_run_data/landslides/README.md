# Landslide detection training labels

This directory contains data for generating training labels for landslide detection models.

## Overview

The pipeline processes landslide inventory data from the [Sen12Landslides dataset](https://www.nature.com/articles/s41597-025-06167-2) to create balanced training labels with positive samples and negative samples.

## Usage

### Step 0: Download inventory data

Download the landslide inventory data from Google Cloud Storage:

```bash
gcloud storage cp gs://rslearn-eai/artifacts/landslides/Sen12Landslides/inventories.zip \
  olmoearth_run_data/landslides/inventories.zip
```

### Step 1: Generate labels

Generate training labels from landslide inventory data:

```bash
uv run python olmoearth_projects/projects/landslides/prepare_labels.py \
  --input-zip olmoearth_run_data/landslides/inventories.zip
```

This creates `landslide_labels.json` with:
- Positive samples: landslide polygons with event dates
- Negative samples: ring buffers around landslide polygons
- Overlap removal: portions of negative rings that overlap with other landslides are removed
- Geographic and label balancing
- Bounding box metrics and task geometries

### Step 2: Create annotation features

Convert the labels to OlmoEarth annotation format:

```bash
uv run python scripts/oer_annotation_creation.py \
  olmoearth_run_data/landslides/landslide_labels.json \
  --taskgeom-col task_geom \
  --outdir olmoearth_run_data/landslides
```

This creates:
- `annotation_task_features.geojson`: Task geometries for annotation
- `annotation_features.geojson`: Annotation features with labels
