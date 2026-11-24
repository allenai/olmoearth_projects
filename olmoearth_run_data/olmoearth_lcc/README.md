## OlmoEarth LCC: Land Cover Change

OlmoEarth LCC detects recent land cover change from Sentinel-2 time series. It inputs
16 quarterly mosaics (a ~4 year historical baseline) followed by 6 weekly mosaics
covering the most recent 91 days, and predicts per-pixel change. The model code, dataset
preparation, and training live in rslearn_projects under `rslp/olmoearth_lcc/`; this
directory holds the olmoearth_run (OlmoEarth Studio) inference config.

### Versions

#### 20261122_rslearn_bpcat_notemporal_unified

- Source configs in rslearn_projects (`data/olmoearth_lcc/lcc_model/`, branch
  `favyen/20260407-change-finder` at `e6b7dc8f`):
  - Training: `config_rslearn_bpcat_notemporal.yaml` (rslearn_projects run
    `2026_11_22_lcc/rslearn_bpcat_notemporal`, trained on
    `/weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_dataset_20261005/`;
    best checkpoint at epoch 58 by `val_post_change/accuracy`). This is the rslearn
    counterpart of the run-17 prediction pipeline model (`bp_abs_bpcat_1pass`), trained on
    the same data.
  - The model block (encoder, the seven decoder stacks and tasks) comes from
    `config_rslearn_bpcat_notemporal_predict.yaml`, the rslearn-only prediction config (no
    `rslp` imports), and the Sentinel-2 windows from `config_predict_rslearn.json`.
- Checkpoint:
  - WEKA: `/weka/dfive-default/rslearn-eai/projects/2026_11_22_lcc/rslearn_bpcat_notemporal/best.ckpt`
  - GCS copy for the platform: `gs://ai2-rslearn-projects-data/projects/2026_11_22_lcc/rslearn_bpcat_notemporal/best.ckpt`

The config is a unified `olmoearth_config.yaml`, so Studio writes every head:

- Model: the seven heads are manual tasks with the decoder layers of the rslearn prediction
  config, and the encoder (OlmoEarth v1.2 base, patch size 4) keeps a token per timestep
  (`token_pooling: false`). The checkpoint loads strictly into the model built from
  `transpiled/model.yaml`.
- Inputs: a `framed_point_in_time` temporality with two lookbehind windows, one dataset
  layer each, fed to the model in order (22 images, as in training):
  - 16 quarterly mosaics from the 1800 days ending 90 days before R;
  - 6 weekly mosaics from the last 91 days (7-day periods can't span exactly 90).

  Training imagery came from Planetary Computer; this config uses the olmoearth_run default
  providers (AWS COGs first), which are pre-harmonized to match.
- Output: a 7-band uint16 raster, one band per head in this order: `binary`,
  `pre_change`, `post_change`, `src`, `dst`, `ts_start`, `ts_end`. The first five are
  argmax classes (below); nodata and "none" are transparent, and the other colors match
  Studio's land change map. `ts_start` and `ts_end` are `per_pixel_date` fields: the
  midpoint of the chosen input image's period as days since 1970-01-01, so a date is
  only as precise as that mosaic: about ±45 days for the quarterly images, ±3 days for
  the weekly ones.
- Vector output: a `vectorize` step polygonizes `post_change`, skipping 0 (nodata) and
  1 ("none"), so each prediction also returns a GeoJSON of the changes. Each polygon also
  carries the other heads as properties: the most common `binary`, `pre_change`, `src`
  and `dst` class under it, and the 10th-percentile `ts_start` and 90th-percentile
  `ts_end` date, written as ISO dates (`YYYY-MM-DD`).
- Requires an olmoearth_run release with `token_pooling` and manual-task heads in
  multi-field models (shared #211, run #643), config version 0.15.0 and `per_pixel_date`
  fields (shared #218, #220, #221; run #613, #659), and rslearn >= 0.1.19 for
  `PerPixelTimestepTask` with date outputs and the `ts_end` head's
  `fallback_to_closest_timestep_if_no_match`. Vectorizing one field of several needs
  shared #198 in both olmoearth_run and Studio (studio #3560); an older Studio silently
  drops `vectorize.field`, and the prediction then fails.

Classes (value = position in the list):

| head | classes |
|---|---|
| `binary` | nodata, no_change, change |
| `pre_change` | nodata, none, forest_loss, removed_building, wetland_loss, water_to_land, removed_crop_structure, mangrove_loss |
| `post_change` | nodata, none, vegetation_growth, new_building, new_road, new_infrastructure, new_agriculture, new_aquafarm, site_clearing, land_to_water, new_mining_or_quarry, new_crop_structure, selective_logging, landslide, settlement, agricultural_activity, burned, flooding, storm_damage, vegetation_dieback |
| `src`, `dst` | nodata, bare, agriculture, grassland, shrub, snow/ice, tree, urban/built-up, water, wetland, mangrove |

`transpiled/` holds what olmoearth_run turns `olmoearth_config.yaml` into for a
prediction, for review: `dataset_config.json` (the rslearn dataset config),
`model.yaml` (the rslearn model config) and `inference_results_config.json` (the
`inference_results_config` part of a legacy `olmoearth_run.yaml`). olmoearth_run
generates these itself at run time. They were generated from olmoearth_monorepo develop
at `e4ae57ce` (`services/run`, rslearn 0.1.19) on a CPU, so `autocast_dtype` reads
`float32`; olmoearth_run sets bfloat16 on a GPU. `<dataset_path>` and
`<model_stage_root>` stand for the per-prediction paths olmoearth_run fills in. The
prediction batch size comes from the deployment (`PREDICTION_BATCH_SIZE`), not the
config; with 22 timesteps and unpooled tokens, check that it fits in GPU memory.

### Registering in Studio

Open an existing LCC model's "Clone model from ingredients", switch the config to
Unified, paste `olmoearth_config.yaml`, set the checkpoint file path to the GCS
checkpoint above, and pick a container image with the release above. Run copies the
checkpoint into the new model's stage.

### Training in Studio

To train it in Studio instead, paste `olmoearth_config.yaml` into the training wizard's
advanced config. The config also carries what training needs: `nodata_value: 255` on the
class tasks (the value olmoearth_run fills unlabeled pixels with), `val_loss` as the
checkpoint metric, and Studio's guided `labeled_data_prep` and `training` defaults for
64-pixel polygon labels. The project has to match it:

- Seven annotation fields named exactly `binary`, `pre_change`, `post_change`, `src`,
  `dst`, `ts_start` and `ts_end`.
- The five class fields are labelsets. Studio exports a label as its position in the
  labelset, ordered by when the label was created, so create each labelset's labels in
  the order of the table above, including `nodata` (and `none`) at the start.
  Nothing checks this; a different order trains pixels as the wrong class.
- `ts_start` and `ts_end` are number fields holding whole days since 1970-01-01. Leave
  them empty where there is no change.
- Each annotation's date is the reference date R, since both image windows look back
  from it.

Training needs an olmoearth_run release with run #651 and #669, and a Studio release with
studio #3343.

### Prediction Request Geometry

The model predicts change as of a reference date R. Set `oe_start_time` on each
feature to R; `oe_end_time` is ignored (set it equal to `oe_start_time`). The quarterly
window reaches back 1890 days (about 5.2 years) before R, so R needs Sentinel-2 L2A
coverage that far back for the region (roughly 2017 onwards), and the weekly window
needs recent imagery up to R.

Windows are 2048x2048 pixels at 10 m in the local UTM zone; the request geometry is
partitioned into 1x1 degree cells first.
