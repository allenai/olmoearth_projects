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
Finally - we treat this as a segmentation task, not as a classification task (this makes inference faster, without hurting performance). This means the point labels need to be transformed into rasters:

```
python olmoearth_projects/projects/mozambique_lulc/create_label_raster.py --ds_path $DATASET_PATH
```

Within `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc` there are four versions of the data:
- `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251023`, which only has the train and test split as defined in the gpkg files
- `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251113`, which splits the training data into train and val data using a spatial split (introduced in [this commit](https://github.com/allenai/olmoearth_projects/pull/28/commits/1cfb86d40c8e2ccba830eb80410d1248544877c9)). This leads to the following train / val / test splits (with `val_ratio = 0.2`):
    - Gaza: 1,802 / 460 / 970
	- Manica: 1,564 / 353 / 822
	- Zambezia: 949 / 276 / 525
	- For crop type mapping, the following train / val / test splits, per class: `'corn': {'train': 917, 'val': 191, 'test': 3709}, 'sesame': {'train': 384, 'val': 0, 'test': 383}, 'beans': {'train': 932, 'val': 224, 'test': 417}, 'rice': {'train': 648, 'val': 512, 'test': 863}, 'millet': {'train': 36, 'val': 0, 'test': 57}, 'cassava': {'train': 685, 'val': 133, 'test': 201}, 'sorghum': {'train': 52, 'val': 0, 'test': 41},`
- `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251114` which aligns the dates for all provinces (as in [this commit](https://github.com/allenai/olmoearth_projects/pull/28/commits/07ee7ef22a383b2c71ef6acab3171df8387924bd)).
- `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251202`, which aligns the dates and the `dataset.json` & `config.json` so that 8 months of data are exported. We also update the val ratio to 0.1 to yield the following splits:
    - crop type mapping: `'corn': {'train': 917, 'val': 191, 'test': 3709}, 'sesame': {'train': 384, 'val': 0, 'test': 383}, 'beans': {'train': 932, 'val': 224, 'test': 417}, 'rice': {'train': 648, 'val': 512, 'test': 863}, 'millet': {'train': 36, 'val': 0, 'test': 57}, 'cassava': {'train': 685, 'val': 133, 'test': 201}, 'sorghum': {'train': 52, 'val': 0, 'test': 41},`
	- LULC: `{'Trees': {'train': 479, 'val': 56, 'test': 229}, 'Cropland': {'train': 1355, 'val': 159, 'test': 649}, 'Buildings': {'train': 858, 'val': 89, 'test': 406}, 'Bare Ground': {'train': 619, 'val': 50, 'test': 288}, 'Water': {'train': 556, 'val': 55, 'test': 263}, 'Rangeland': {'train': 514, 'val': 57, 'test': 245}, 'Flooded Vegetation': {'train': 500, 'val': 57, 'test': 237}`.

#### Assessing label quality

Label quality can be assessed by running the `check_label_quality.py` script:

```console
$ python olmoearth_projects/projects/mozambique_lulc/check_label_quality.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251202 --split train

Checking label quality for 4881 instances.
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃         Check name ┃ Metric                ┃               Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│    label_imbalance │ Bare Ground           │ 0.12681827494365908 │
│    label_imbalance │ Trees                 │ 0.09813562794509322 │
│    label_imbalance │ Cropland              │ 0.27760704773611966 │
│    label_imbalance │ Flooded Vegetation    │  0.1024380249948781 │
│    label_imbalance │ Water                 │ 0.11391108379430445 │
│    label_imbalance │ Rangeland             │ 0.10530628969473468 │
│    label_imbalance │ Buildings             │  0.1757836508912108 │
│ spatial_clustering │ Bare Ground_f1        │  0.7763055339049103 │
│ spatial_clustering │ Trees_f1              │               0.918 │
│ spatial_clustering │ Cropland_f1           │  0.8201489890031926 │
│ spatial_clustering │ Flooded Vegetation_f1 │  0.6470588235294118 │
│ spatial_clustering │ Water_f1              │  0.5609756097560976 │
│ spatial_clustering │ Rangeland_f1          │  0.7097480832420592 │
│ spatial_clustering │ Buildings_f1          │  0.9638554216867469 │
│     spatial_extent │ Bare Ground           │   0.906388431021162 │
│     spatial_extent │ Trees                 │  0.8143211426450099 │
│     spatial_extent │ Cropland              │  0.8178565572914295 │
│     spatial_extent │ Flooded Vegetation    │  0.8195186876112993 │
│     spatial_extent │ Water                 │  0.8015534585021155 │
│     spatial_extent │ Rangeland             │  0.9892764881988351 │
│     spatial_extent │ Buildings             │  0.7256137393021044 │
└────────────────────┴───────────────────────┴─────────────────────┘
```
and for crop type:
```console
$ python olmoearth_projects/projects/mozambique_lulc/check_label_quality.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251202 --split train --crop_type

Checking label quality for 3821 instances.
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃         Check name ┃ Metric     ┃                 Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│    label_imbalance │ beans      │     0.260664747448312 │
│    label_imbalance │ corn       │   0.24522376341271918 │
│    label_imbalance │ cassava    │   0.18503009683328972 │
│    label_imbalance │ sesame     │   0.10049725202826486 │
│    label_imbalance │ rice       │   0.18555352002093692 │
│    label_imbalance │ sorghum    │  0.013609002878827532 │
│    label_imbalance │ millet     │   0.00942161737764983 │
│ spatial_clustering │ beans_f1   │    0.9885401096163426 │
│ spatial_clustering │ corn_f1    │    0.9946581196581196 │
│ spatial_clustering │ cassava_f1 │    0.9728571428571429 │
│ spatial_clustering │ sesame_f1  │                   1.0 │
│ spatial_clustering │ rice_f1    │    0.9943661971830987 │
│ spatial_clustering │ sorghum_f1 │    0.9902912621359223 │
│ spatial_clustering │ millet_f1  │                   1.0 │
│     spatial_extent │ beans      │    0.7829124125842517 │
│     spatial_extent │ corn       │    0.8357589381957512 │
│     spatial_extent │ cassava    │    0.9383923435655623 │
│     spatial_extent │ sesame     │ 0.0003614488654102921 │
│     spatial_extent │ rice       │    0.7653946614854196 │
│     spatial_extent │ sorghum    │ 3.266172530759744e-07 │
│     spatial_extent │ millet     │ 0.0001509740414169792 │
└────────────────────┴────────────┴───────────────────────┘
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
