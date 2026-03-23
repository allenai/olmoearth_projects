### Ethiopia maize

This script demonstrates how to prepare shapefiles for OlmoEarth.
These labels have the following properties:

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃         Check name ┃ Metric       ┃               Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│        # instances │              │               24673 │
│    label_imbalance │ maize        │ 0.48417298261257247 │
│    label_imbalance │ non_maize    │  0.5158270173874275 │
│ spatial_clustering │ maize_f1     │   0.780725806451613 │
│ spatial_clustering │ non_maize_f1 │  0.7784567750346288 │
│     spatial_extent │ maize        │  0.8004380118991101 │
│     spatial_extent │ non_maize    │  0.9904858160033082 │
└────────────────────┴──────────────┴─────────────────────┘

OlmoEarth accepts geojsons (or csvs) with the following columns:

1. `start_date`
2. `end_date`: we will pair each labels with observations between `start_date` and `end_date`
3. `geometry`
4. A label column (in this case `maize_or_not`)

Once this geojson has been added to Studio, we can finetune a model on it.
An example unified config used for finetuning and inference is stored here ([`unified_config.yaml`](unified_config.yaml)).
