# Nigeria Maize vs. not Maize classification

### Generating the data
```
python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/nigeria_maize/create_training_points.py --label_dir /weka/dfive-default/gabrielt/datasets/nigeria_maize

export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/nigeria_maize/20251125

python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/nigeria_maize/create_windows.py --geojson_file /weka/dfive-default/gabrielt/datasets/nigeria_maize/combined.geojson --ds_path $DATASET_PATH --window_size 32
```
You will then need to copy a config.json into $DATASET_PATH.

The config being used is available in [config.json](config.json).

Once the config is copied into the dataset root, the following commands can be run:

```
rslearn dataset prepare --root $DATASET_PATH --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

python -m rslp.main common launch_data_materialization_jobs --image yawenzzzz/rslp20251112h --ds_path $DATASET_PATH --clusters+=ai2/neptune-cirrascale --num_jobs 5
```
