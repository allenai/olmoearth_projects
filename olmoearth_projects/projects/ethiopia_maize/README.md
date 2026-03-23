### Ethiopia maize

This script demonstrates how to prepare shapefiles for OlmoEarth.

OlmoEarth accepts geojsons (or csvs) with the following columns:

1. `start_date`
2. `end_date`: we will pair each labels with observations between `start_date` and `end_date`
3. `geometry`
4. A label column (in this case `maize_or_not`)

Once this geojson has been added to Studio, we can finetune a model on it.
An example unified config used for finetuning and inference is stored here ([`unified_config.yaml`](unified_config.yaml)).
