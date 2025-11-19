# Mozambique LULC and Crop Type Classification

This project has two main tasks:
	1.	Land Use/Land Cover (LULC) and cropland classification
	2.	Crop type classification

The annotations come from field surveys across three provinces in Mozambique: Gaza, Zambezia, and Manica.

For LULC classification, the train/test splits are:
- Gaza: 2,262 / 970
- Manica: 1,917 / 822
- Zambezia: 1,225 / 525

### Generating the data
```
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251113

python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/mozambique_lulc/create_windows_for_lulc.py --gpkg_dir /weka/dfive-default/yawenz/datasets/mozambique/train_test_samples --ds_path $DATASET_PATH --window_size 32

python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/mozambique_lulc/create_windows_for_lulc.py --gpkg_dir /weka/dfive-default/yawenz/datasets/mozambique/train_test_samples --ds_path $DATASET_PATH --window_size 32 --crop_type
```
You will then need to copy a `config.json` into `$DATASET_PATH`.

The config being used is available in [config.json](config.json). This config requires [rslearn_projects](https://github.com/allenai/rslearn_projects) in your environment.

Once the config is copied into the dataset root, the following commands can be run:

```
rslearn dataset prepare --root $DATASET_PATH --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

python -m rslp.main common launch_data_materialization_jobs --image yawenzzzz/rslp20251112h --ds_path $DATASET_PATH --clusters+=ai2/neptune-cirrascale --num_jobs 5
```

Within `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc` there are two versions of the data:
- `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251023`, which only has the train and test split as defined in the gpkg files
- `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251113`, which splits the training data into train and val data using a spatial split (introduced in [this commit](https://github.com/allenai/olmoearth_projects/pull/28/commits/1cfb86d40c8e2ccba830eb80410d1248544877c9)). This leads to the following train / val / test splits (with `val_ratio = 0.2`):
    - Gaza: 1,802 / 460 / 970
	- Manica: 1,564 / 353 / 822
	- Zambezia: 949 / 276 / 525
	- For crop type mapping, the following train / val / test splits, per class: `'corn': {'train': 917, 'val': 191, 'test': 3709}, 'sesame': {'train': 384, 'val': 0, 'test': 383}, 'beans': {'train': 932, 'val': 224, 'test': 417}, 'rice': {'train': 648, 'val': 512, 'test': 863}, 'millet': {'train': 36, 'val': 0, 'test': 57}, 'cassava': {'train': 685, 'val': 133, 'test': 201}, 'sorghum': {'train': 52, 'val': 0, 'test': 41},`
- `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251114` which aligns the dates for all provinces (as in [this commit](https://github.com/allenai/olmoearth_projects/pull/28/commits/07ee7ef22a383b2c71ef6acab3171df8387924bd)).

Finally - we treat this as a segmentation task, not as a classification task (this makes inference faster, without hurting performance). This means the point labels need to be transformed into rasters:

```
python olmoearth_projects/projects/mozambique_lulc/create_label_raster.py --ds_path $DATASET_PATH
```

### Finetuning

Currently, we use [rslearn_projects](github.com/allenai/rslearn_projects) for finetuning, using [rslp_finetuning.yaml](rslp_finetuning.yaml) and [rslp_finetuning_croptype.yaml](rslp_finetuning_croptype.yaml).  With `rslean_projects` installed (and access to Beaker), finetuning can then be run with the following command:

```
python -m rslp.main olmoearth_pretrain launch_finetune --image_name yawenzzzz/rslp20251112h --config_paths+=olmoearth_run_data/mozambique_lulc/rslp_finetuning.yaml --cluster+=ai2/saturn --rslp_project <MY_RSLP_PROJECT_NAME> --experiment_id <MY_EXPERIMENT_ID>
```

### Testing

Obtaining test results consisted of the following:
1. Spin up an interactive beaker session with a GPU: `beaker session create --remote --bare --budget ai2/es-platform --cluster ai2/saturn --mount src=weka,ref=dfive-default,dst=/weka/dfive-default --image beaker://yawenzzzz/rslp20251112h --gpus 1`
2. Go to the olmoearth projects folder on weka (to easily `git pull`) changes: `cd /weka/dfive-default/gabrielt/olmoearth_projects`
3. Add the `RSLP_PREFIX` to the environment, `export RSLP_PREFIX=/weka/dfive-default/rslearn-eai`
4. Run testing: `python -m rslp.rslearn_main model test --config olmoearth_run_data/mozambique_lulc/rslp_finetuning.yaml --rslp_experiment <MY_EXPERIMENT_ID> --rslp_project <MY_RSLP_PROJECT_NAME> --force_log=true --load_best=true --verbose true`

### Inference

All inference is done on [OlmoEarth Studio](https://olmoearth.allenai.org/). Polygons around the provinces were manually drawn (within Studio).
