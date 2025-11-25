### Step 7: Train the Model

```bash
rslearn model fit --config dataset_floods/model.yaml
```

Note: make sure to have dataset_floods/model.yaml

```bash
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_BASE
              patch_size: 4
        decoder:
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              in_channels: [[4, 768]]
              out_channels: 2
              conv_layers_per_resolution: 2
              num_channels: {4: 512, 2: 256, 1: 128}
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
    optimizer:
      class_path: rslearn.train.optimizer.AdamW
      init_args:
        lr: 0.0001

data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ./dataset_floods
    inputs:
      sentinel1:
        data_type: raster
        layers: ["sentinel1"]
        bands: ["vv", "vh"]
        dtype: FLOAT32
        passthrough: true
      sentinel2_l2a:
        data_type: raster
        layers: ["sentinel2_l2a"]
        bands: ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
        dtype: FLOAT32
        passthrough: true
        load_all_layers: true
      targets:
        data_type: raster
        layers: ["label"]
        bands: ["B1"]
        dtype: FLOAT32
        is_target: true
    task:
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        num_classes: 2
        enable_miou_metric: true
        nodata_value: 255
    batch_size: 4
    num_workers: 32
    default_config:
      groups: ["default"]
      patch_size: 128
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
              sentinel1: ["vv", "vh"]
    train_config:
      tags:
        split: "train"
    val_config:
      tags:
        split: "val"
    predict_config:
      groups: ["predict"]
      patch_size: 512 
      skip_targets: true 
trainer:
  max_epochs: 100
  strategy: ddp_find_unused_parameters_true
  logger:
    class_path: lightning.pytorch.loggers.CSVLogger
    init_args:
      save_dir: ./logs
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: ./checkpoints
        save_top_k: 1
        save_last: true
        monitor: val_mean_iou
        mode: max
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: ["model", "encoder", 0]
        unfreeze_at_epoch: 10
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        # This path will be copied from data.init_args.path by rslearn.
        path: placeholder
        output_layer: output
        merger:
          class_path: rslearn.train.prediction_writer.RasterMerger
```