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
python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/mozambique_lulc/create_windows_for_lulc.py --gpkg_dir /weka/dfive-default/yawenz/datasets/mozambique/train_test_samples --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251113 --window_size 32
```
You will then need to copy a `config.json` into the dataset path, `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251113`.

The config being used is available in [config.json](config.json). This config requires [rslearn_projects](https://github.com/allenai/rslearn_projects) in your environment.

Once the config is copied into the dataset root, the following commands can be run:

```
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251113 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

python -m rslp.main common launch_data_materialization_jobs --image yawenzzzz/rslp20251112h --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251113 --clusters+=ai2/neptune-cirrascale --num_jobs 5
```

Within `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc` there are two versions of the data:
- `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251023`, which only has the train and test split as defined in the gpkg files
- `/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251113`, which splits the training data into train and val data using a spatial split (introduced in [this commit](https://github.com/allenai/olmoearth_projects/pull/28/commits/1cfb86d40c8e2ccba830eb80410d1248544877c9)). This leads to the following train / val / test splits (with `val_ratio = 0.2`):
    - Gaza: 1,802 / 460 / 970
	- Manica: 1,564 / 353 / 822
	- Zambezia: 949 / 276 / 525

Finally - we treat this as a segmentation task, not as a classification task (this makes inference faster, without hurting performance). This means the point labels need to be transformed into rasters:

```
python olmoearth_projects/projects/mozambique_lulc/create_label_raster.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251113
```

### Finetuning

Currently, we use [rslearn_projects](github.com/allenai/rslearn_projects) for finetuning, using the [rslp_finetuning.yaml](rslp_finetuning.yaml). To run finetune for a specific province, update the yaml's `groups` (lines 238-250) from `"gaza"` to one of `["gaza", "manica", "zambezia"]`. With `rslean_projects` installed (and access to Beaker), finetuning can then be run with the following command:

```
python -m rslp.main olmoearth_pretrain launch_finetune --image_name gabrielt/rslpomp_20251027b --config_paths+=olmoearth_run_data/mozambique_lulc/rslp_finetuning.yaml --cluster+=ai2/saturn --rslp_project <MY_RSLP_PROJECT_NAME> --experiment_id <MY_EXPERIMENT_ID>
```

### Testing

Obtaining test results consisted of the following:
1. Spin up an interactive beaker session with a GPU: `beaker session create --remote --bare --budget ai2/es-platform --cluster ai2/saturn --mount src=weka,ref=dfive-default,dst=/weka/dfive-default --image beaker://gabrielt/rslpomp_20251027b --gpus 1`
2. Go to the olmoearth projects folder on weka (to easily `git pull`) changes: `cd /weka/dfive-default/gabrielt/olmoearth_projects`
3. Run testing: `python -m rslp.rslearn_main model test --config olmoearth_run_data/mozambique_lulc/rslp_finetuning.yaml --rslp_experiment gaza_with_val_20251114_ps1 --rslp_project 2025_09_18_mozambique_lulc --force_log=true --load_best=true --verbose true`

I used confusion matrices to combine the results for each province into a single set of matrices. Getting these from pytorch lightning proved quite messy.

I added the following to the config:
```
                confusion_matrix:
                  class_path: rslearn.train.tasks.segmentation.SegmentationMetric
                  init_args:
                    metric:
                      class_path: torchmetrics.ConfusionMatrix
                      init_args:
                        num_classes: 8
                        ignore_index: 0
                        task: "multiclass"
                    class_idx: null
```

But pytorch lightning only wants to return scalar metrics, so the following error gets thrown:
```
pytorch/trainer/connectors/logger_connector/logger_connector.py", line 106, in log_metrics
    scalar_metrics = convert_tensors_to_scalars(metrics)
```
If you just add `print (metrics)` before `convert_tensors_to_scalars` in `logger_connector.py`, you can get the confusion matrix! (Alternatively the error also prints out the full confusion matrix, so you don't even need to add the print statement, just let it error).

### Inference

All inference is done on [OlmoEarth Studio](https://olmoearth.allenai.org/). Polygons around the provinces were manually drawn (within Studio).
