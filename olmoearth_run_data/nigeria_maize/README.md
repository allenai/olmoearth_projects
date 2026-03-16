# Nigeria Maize vs. not Maize classification

### Generating the data
```
python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/nigeria_maize/create_training_points.py --label_dir /weka/dfive-default/gabrielt/datasets/nigeria_maize

export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/nigeria_maize/20251126

python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/nigeria_maize/create_windows.py --geojson_file /weka/dfive-default/gabrielt/datasets/nigeria_maize/combined_labels.geojson --ds_path $DATASET_PATH --window_size 32
```
You will then need to copy a config.json into $DATASET_PATH.

The config being used is available in [config.json](config.json).

Once the config is copied into the dataset root, the following commands can be run:

```
rslearn dataset prepare --root $DATASET_PATH --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

python -m rslp.main common launch_data_materialization_jobs --image yawenzzzz/rslp20251112h --ds_path $DATASET_PATH --clusters+=ai2/neptune-cirrascale --num_jobs 5
```
Finally - we treat this as a segmentation task, not as a classification task (this makes inference faster, without hurting performance). This means the point labels need to be transformed into rasters:

```
python olmoearth_projects/projects/nigeria_maize/create_label_raster.py --ds_path $DATASET_PATH
```
### Finetuning

Currently, we use [rslearn_projects](github.com/allenai/rslearn_projects) for finetuning, using [rslp_finetuning.yaml](rslp_finetuning.yaml) and [rslp_finetuning_croptype.yaml](rslp_finetuning_croptype.yaml).  With `rslean_projects` installed (and access to Beaker), finetuning can then be run with the following command:

```
python -m rslp.main olmoearth_pretrain launch_finetune --image_name yawenzzzz/rslp20251112h --config_paths+=olmoearth_run_data/nigeria_maize/rslp_finetuning.yaml --cluster+=ai2/saturn --rslp_project <MY_RSLP_PROJECT_NAME> --experiment_id <MY_EXPERIMENT_ID>
```

### Progress

#### 2025-11-26
Add a new dataset for negative samples (`https://rdm.esa-worldcereal.org/collections/2021_glo_ewocval_poly_111`).

#### 2025-11-25
Initial run of the model. Seems to overpredict maize?
