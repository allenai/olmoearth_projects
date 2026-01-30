# Togo cropland classification

This project consists of cropland prediction in Togo. We use the labels from [github.com/nasaharvest/togo-crop-mask](github.com/nasaharvest/togo-crop-mask).

```python
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃         Check name ┃ Metric   ┃               Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│        # instances │          │                1582 │
│    label_imbalance │ True     │  0.5018963337547409 │
│    label_imbalance │ False    │ 0.49810366624525915 │
│ spatial_clustering │ True_f1  │  0.6768687463213656 │
│ spatial_clustering │ False_f1 │  0.6252559726962458 │
│     spatial_extent │ True     │  0.9920167295161375 │
│     spatial_extent │ False    │   0.999596043028759 │
└────────────────────┴──────────┴─────────────────────┘
```
### Generating the data\

Download the labels from [https://zenodo.org/records/3836629](https://zenodo.org/records/3836629), and store them in `--shapefile_dir`.
```bash
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/togo_2020/20260127

python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/togo_cropland/create_windows.py --shapefile_dir /weka/dfive-default/gabrielt/datasets/togo_labels --ds_path $DATASET_PATH --window_size 32
```
You will then need to copy a `config.json` into `$DATASET_PATH`. The config being used is available in [config.json](config.json).

Once the config is copied into the dataset root, the following commands can be run:

```bash
rslearn dataset prepare --root $DATASET_PATH --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

python -m rslp.main common launch_data_materialization_jobs --image yawenzzzz/rslp20251112h --ds_path $DATASET_PATH --clusters+=ai2/neptune-cirrascale --num_jobs 5
```
Finally - we treat this as a segmentation task, not as a classification task (this makes inference faster, without hurting performance). This means the point labels need to be transformed into rasters:

```
python olmoearth_projects/projects/togo_cropland/create_label_raster.py --ds_path $DATASET_PATH
```

### Finetuning
Currently, we use [rslearn_projects](https://github.com/allenai/rslearn_projects) for finetuning, using [rslp_finetuning.yaml](rslp_finetuning.yaml). With rslean_projects installed (and access to Beaker), finetuning can then be run with the following command:

```
python -m rslp.main olmoearth_pretrain launch_finetune \
--image_name gabrielt/20260113_rslpomp \
--config_paths+=olmoearth_run_data/togo_cropland/rslp_finetuning.yaml \
--cluster+=ai2/saturn \
--rslp_project <MY_RSLP_PROJECT_NAME> \
--experiment_id <MY_EXPERIMENT_ID>
```
