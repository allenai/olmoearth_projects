# Finetuning OlmoEarth on a classification task - Example

| # | Section |
| - | - |
| 0 | [Setup](#1-setup) |
| 1 | [Setup](#1-setup) |
| 2 | [Gathering and prepping the dataset](#2-Gathering-and-prepping-the-dataset) |
| 3 | [Benchmarking the base model](#3-benchmarking-the-base-model) |
| 4 | [Defining your ...](#4-defining-your-) |
| 5 | [Training](#5-training) |
| 6 | [Using your fine-tuned model](#6-using-your-fine-tuned-model) |


## 0. Goal
Let's build a burned area detection model using OlmoEarth. We will finetune the base model on a burned area mapping task and use it to detect fire perimeters on a unseen past fires.


## 1. Setup


## 2. Gathering and prepping the dataset
Letʼs start off by using fire perimeter data from CalFire. These come as polygons associated with a contained date, which will use to determine when our satellite snapshots should be captured (usually within 2 weeks of the contained date).

Let's download the data:
``` shell
curl -L -A "Mozilla/5.0" -e "https://www.fire.ca.gov/" "https://34c031f8-c9fd-4018-8c5a-4159cdff6b0d-cdn-endpoint.azureedge.net/-/media/calfire-website/what-we-do/fire-resource-assessment-program---frap/gis-data/2025/fire241gdb.ashx?rev=51177a999fe84e83a7c03b7d5a66b93b" -o fire241gdb.zip
```

The data comes in the form of burnt polygons, each associated with a containment date and an alarm date.
We will use the rslearn library to process the data. This library makes it convenient to load and store the data in a structured format that can be easily consumed by OlmoEarth model.

> ---
> **TODO — Document The pre-processing stage -> negative point creation + sampling**
> - Negative sampling strategy -> outer ring with inner ring buffer around fire perimeter. Need to improve with other non-fire-associated locations
> - point sampling in polygon: Grid approach. Need to improve: number of allocation points within polygon should be total area dependent to avoid multiple points per window
> ---

### Preparing a dataset using the Rslearn library
An rslearn *dataset* consists of a set of raster and vector layers, along with
spatiotemporal windows where data in those layers is available. A *window* roughly corresponds to a training or test example, and is the unit for both data annotation and model training. Essentially, a window is a geographic area, coupled with a time range, over which we want a model to make some prediction.

#### Window creation
--------
Here we are creating the windows at the given sampled locations:
``` shell
python ./rslearn_projects/rslp/burn-scar/create_windows_for_groundtruth.py --csv_path=/weka/dfive-default/hadriens/datasets/label_data/Calfp_2020-2025_GroundTruthPoints.csv --ds_path=/weka/dfive-default/rslearn-eai/datasets/burn-scar/California/2020-25 --window_size=32
```
This effectively stores windows in the ds_path folder. We used a window_size of 32 and a window projection of 10 m/pixel (x) and −10 m/pixel (y), so a 32-pixel window is ~320 m × 320 m on the ground. 


> ---
> **Additional info**
> - The dataloader reads exactly the inputs you specify (e.g., Sentinel-2) cropped to that window (or a patch inside it, if you set patch_size).
It does not pull context from neighboring windows. If you want larger context, increase window size or use a larger patch_size. For sliding prediction, use load_all_patches + patch_size.
> ---



At this point there is one window per data point, and each window is represented by one metadata.json file containing the group, name, bounds, projection and time_range of the window. These stored metadata information allow to create data layers for each window.
Each *layer* stores a certain kind of data. In our case, our dataset currently has a vector layer where burnt and unburnt locations are labeled. The layer is associated with a data.geojson file that encodes the window geometry, as well as the label (burnt or unburnt).

#### Dataset configuration
--------
At this point we need to create a config.json file that defines our dataset schema: which layers exist, their type (raster/vector), formats, and optionally how to auto-populate them via a data_source.
In our case, so far we have created our windows and our label layer in a vector format (each window as a data.geojson file), so we need to reflect this in our config file:



```json
"layers": {
    "label": {
        "type": "vector",
        "format": {
            "name": "geojson", 
            "coordinate_mode": "pixel"
        }
    },
    "sentinel2": {
        "type": "raster",
        "band_sets": [
            {
            "dtype": "uint16",
            "bands": ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
            },
            {
            "dtype": "uint8",
            "bands": ["R","G","B"]
            }
            ],
        "resampling_method": "bilinear",
        "data_source": {
            "name": "rslearn.data_sources.gcp_public_data.Sentinel2",
            "index_cache_dir": "cache/sentinel2/",
            "sort_by": "cloud_cover",
            "use_rtree_index": false
        }
    }
}
```
"coordinate_mode": "pixel" indicate that the vector on disk (layers/{layer}/data.geojson) have coordinates in pixel units.
Mixed native resolutions (10/20/60 m) are fine — rslearn resamples to the window’s resolution during materialize using resampling_method.
"sort_by" controls ordering of candidate scenes before matching; supported value here is "cloud_cover" (ascending).
Use to prefer lowest‑cloud scenes when building mosaics or selecting first valid.
"use_rtree_index" uses a spatial rtree for fast scene lookup (speeds up prepare a lot on big spatiotemporal extents).

> ---
> **Note — time steps**
> - The layer QueryConfig (in the dataset config.json) defaults to: space_mode=MOSAIC, time_mode=WITHIN, max_matches=1. That means it finds items within the window time_range and forms up to one “group.”
> - For multiple time steps, here is an example that fetches the best (wrt cloud_cover if "sort_by": "cloud_cover") window item for every 15d period within the defined window time range: "query_config": {
"space_mode": "PER_PERIOD_MOSAIC",
"time_mode": "WITHIN",
"period_duration": "15d",
"max_matches": 2 #max periods in the time range
}
> ---


> ---
> **TODO — Questions**
> - How is the provided time range handled by RSlearn when downloading/stitching data together? Does the above config specifies to only select one (best wrt cloud coverage)?
> ---

#### Preparing Dataset
--------
Now we will match items in the data source with each window in the dataset. The output of matching is a list of *item groups* for each window, where each group specifies a different list of items that should be mosaiced to form a different sub-layer for that window.

``` shell
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/burn-scar/California/2020-25
export DATASET_GROUP=groundtruth_polygon_split_window_32
rslearn dataset prepare --root $DATASET_PATH --group $DATASET_GROUP --workers 32 --retry-max-attempts 8
```
Notice that after this step, we have created an items.json file for each window. This json lists the different items from the source (here sentinel 2) that intersects with the associated window, along with the cloud cover %.

#### Ingesting Dataset
--------
The next step is to actually download the items (across all windows in the dataset) into the configured tile store for the dataset. The items are converted to formats (e.g. GeoTIFF for raster data or tiled set of GeoJSONs for vector data) that support random access to enable fast materialization. To do so we call the rslearn dataset *ingest* command:

``` shell
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/burn-scar/California/2020-25
export DATASET_GROUP=groundtruth_polygon_split_window_32
rslearn dataset ingest --root $DATASET_PATH --group $DATASET_GROUP --workers 32 --retry-max-attempts 8
```
After this step, we have populated a "tiles" folder within our dataset path. Each item listed in the previous steps have been downloaded, and a geotiff.tif file has been created for each listed band for each item.

#### Materializing Dataset
--------
Now we need to crop, re-project, and mosaic these items to extract portions aligned with the windows. For raster data (sentinel 2 data in our case), this means the source GeoTIFFs are merged and cropped to correspond to the projection and bounds of the window. For vector data,
features would be concatenated across items in the same group, and then items that do not intersect the window bounds would be filtered out.

``` shell
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/burn-scar/California/2020-25
export DATASET_GROUP=groundtruth_polygon_split_window_32
rslearn dataset materialize --root $DATASET_PATH --group $DATASET_GROUP --workers 64 --retry-max-attempts 8
```
We now have additional layer in our dataset windows, named "sentinel2" in accordance with our dataset config.json file. For each window, our band sets (B01-B12 and RGB) have been stiched into two respective geotiffs (one for each band set) that covers the extent of the window at the appropriate resolution.

We now have a dataset consisting of vector ground truth data and raster sentinel 2 covariate data. 
We are ready to finetune OlmoEarth.

## 3. Defining and training the model

First let's download the OlmoEarth pretrained model weights. Depending on the complexity of the task, your finetuning budget, and your GPU memory, you can select:
  - OlmoEarth nano: {X} parameters - {HF link}:
  - OlmoEarth tiny: {X} parameters, - {HF link}:
  - OlmoEarth base: {X} parameters, - {HF link}:
  - OlmoEarth large: {X} parameters, - {HF link}:


Now we need to design our model architecture, training loop and evaluation metrics, and define how the data should be pre-processed and sent to the model.
Behind the scenes, we use Lightning to coordinate and run the fine-tuning job. This allows us to configure every aspect of the job in a single configuration file.

Let's create it: 
`burn_scar_model.yaml`:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.MultiTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              checkpoint_path: /weka/dfive-default/helios/checkpoints/henryh/base_v6.1_add_chm_cdl_worldcereal/step500000
              selector: ["encoder"]
              forward_kwargs:
                patch_size: 4
        decoders:
          burn-scar_classification:
            - class_path: rslearn.models.pooling_decoder.PoolingDecoder
              init_args:
                in_channels: 768
                out_channels: 2
            - class_path: rslearn.train.tasks.classification.ClassificationHead
    lr: 0.0001
    scheduler:
      class_path: rslearn.train.scheduler.PlateauScheduler
      init_args:
        factor: 0.2
        patience: 2
        min_lr: 0
        cooldown: 10
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    # Replace this with the dataset path.
    path: /weka/dfive-default/rslearn-eai/datasets/burn-scar/California/2020-25
    # This defines the layers that should be read for each window.
    # The key ("image" / "targets") is what the data will be called in the model,
    # while the layers option specifies which layers will be read.
    inputs:
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2_l2a"]
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true #if false only what the Task returns via process_inputs is sent to the model. If true the raw input (here is the raster) is added to the Task output, which is then accessible for transforms or visualization
        dtype: FLOAT32
      label:
        data_type: "vector"
        layers: ["label"]
        is_target: true
    task:
      class_path: rslearn.train.tasks.multi_task.MultiTask
      init_args:
        tasks:
          burn-scar_classification:
            class_path: rslearn.train.tasks.classification.ClassificationTask
            init_args:
              property_name: "category"
              classes: ["burnt", "unburnt"]
              enable_f1_metric: true
              metric_kwargs:
                average: "micro"
        input_mapping:
          burn-scar_classification:
            label: "targets"
    batch_size: 32
    num_workers: 32
    # These define different options for different phases/splits, like training,
    # validation, and testing.
    # Here we use the same transform across splits except training where we add a
    # flipping augmentation.
    # For now we are using the same windows for training and validation.
    default_config:
      transforms:
        - class_path: rslearn.train.transforms.concatenate.Concatenate
          init_args:
            selections:
              sentinel2_l2a: []
            output_selector: sentinel2_l2a
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        - class_path: rslearn.train.transforms.pad.Pad
          init_args:
            size: 8
            mode: "center"
            image_selectors: ["sentinel2_l2a"]
    train_config:
      groups: ["groundtruth_polygon_split_window_32"]
      tags:
        split: "train"
    val_config:
      groups: ["groundtruth_polygon_split_window_32"]
      tags:
        split: "val"
    test_config:
      groups: ["groundtruth_polygon_split_window_32"]
      tags:
        split: "val"
trainer:
  max_epochs: 100
  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: "epoch"
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 1
        save_last: true
        monitor: val_loss
        mode: min
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: ["model", "encoder", 0]
        unfreeze_at_epoch: 2
rslp_project: placeholder
rslp_experiment: placeholder
```

Now we can train the model:

```shell
python -m rslp.main olmoearth_pretrain launch_finetune --image_name yawenzzzz/rslpomp --config_paths+=data/helios/v2_burn_scar_classification/finetune_s2_single_ts.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_10_20_burn-scar-ca --experiment_id burn-scar-ca_classif_helios_base_S2_1ts_ws4_ps4_bs32
```

## 4. Comparing different set-up (multimodal, tasks)
## 5. Using your fine-tuned model
