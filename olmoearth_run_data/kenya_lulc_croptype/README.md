# Kenya LULC and Maize Classification

This project has two main tasks:
	1.	Land Use/Land Cover (LULC) and cropland classification
	2.	Maize vs. rest classification

The annotations come from WorldCereal's Reference Data Module, specifically the following 3 datasets:

1. https://rdm.esa-worldcereal.org/collections/2019_ken_nhicropharvest_point_100
2. https://rdm.esa-worldcereal.org/collections/2021_ken_copernicusgeoglamsr_point_111
3. GeoGlam 2023, short rains

In addition, an additonal round of negative (non-crop) labelling was conducted in 2026. The labels were created by running `olmoearth_projects/projects/create_training_points.py`:

```python
>>> import geopandas as gpd
>>> from olmoearth_projects.utils.label_quality import check_label_quality
>>>
>>> labels = gpd.read_file("kenya_labels/labels.geojson")
>>> # to assess the cropland label quality, make the is_crop column the label
>>> labels["label"] = labels.is_crop.astype(str)
>>> check_label_quality(labels)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃         Check name ┃ Metric   ┃              Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│        # instances │          │              13701 │
│    label_imbalance │ False    │ 0.5079921173636961 │
│    label_imbalance │ True     │ 0.4920078826363039 │
│ spatial_clustering │ False_f1 │ 0.7356907401857955 │
│ spatial_clustering │ True_f1  │ 0.7489682652625588 │
│     spatial_extent │ False    │                1.0 │
│     spatial_extent │ True     │ 0.5357648281826433 │
└────────────────────┴──────────┴────────────────────┘
>>> # to assess the maize label quality, make the maize column the label
>>> labels.label = (labels.sampling_ewoc_code == "maize").astype(str)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃         Check name ┃ Metric   ┃               Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│        # instances │          │               13701 │
│    label_imbalance │ False    │  0.7952704182176483 │
│    label_imbalance │ True     │ 0.20472958178235165 │
│ spatial_clustering │ False_f1 │  0.8787162915088031 │
│ spatial_clustering │ True_f1  │ 0.45218441715321117 │
│     spatial_extent │ False    │                 1.0 │
│     spatial_extent │ True     │  0.5296822018369517 │
└────────────────────┴──────────┴─────────────────────┘
```
### Generating the data
```bash
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_maize_cropland/20260126

python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/kenya_lulc_croptype/create_windows.py --geojson_file /weka/dfive-default/gabrielt/datasets/kenya_labels/labels.geojson --ds_path $DATASET_PATH --window_size 32
```
You will then need to copy a `config.json` into `$DATASET_PATH`. The config being used is available in [config.json](config.json).

Once the config is copied into the dataset root, the following commands can be run:

```bash
rslearn dataset prepare --root $DATASET_PATH --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

python -m rslp.main common launch_data_materialization_jobs --image yawenzzzz/rslp20251112h --ds_path $DATASET_PATH --clusters+=ai2/neptune-cirrascale --num_jobs 5
```
Finally - we treat this as a segmentation task, not as a classification task (this makes inference faster, without hurting performance). This means the point labels need to be transformed into rasters:

```
python olmoearth_projects/projects/kenya_lulc_croptype/create_label_raster.py --ds_path $DATASET_PATH
```

### Finetuning
Currently, we use [rslearn_projects](https://github.com/allenai/rslearn_projects) for finetuning, using [rslp_finetuning_cropland.yaml](rslp_finetuning_cropland.yaml) and [rslp_finetuning_maize.yaml]. With rslean_projects installed (and access to Beaker), finetuning can then be run with the following command:

```
python -m rslp.main olmoearth_pretrain launch_finetune \
--image_name gabrielt/20260113_rslpomp \
--config_paths+=olmoearth_run_data/mozambique_lulc/rslp_finetuning.yaml \
--cluster+=ai2/saturn \
--rslp_project <MY_RSLP_PROJECT_NAME> \
--experiment_id <MY_EXPERIMENT_ID>
```
