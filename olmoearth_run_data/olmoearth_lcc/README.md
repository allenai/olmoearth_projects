## OlmoEarth LCC: Land Cover Change

OlmoEarth LCC detects recent land cover change from Sentinel-2 time series. It inputs
16 quarterly mosaics (a ~4 year historical baseline) followed by 4 biweekly mosaics
covering the most recent 60 days, and predicts per-pixel change. The model code, dataset
preparation, and training live in rslearn_projects under `rslp/olmoearth_lcc/`; this
directory holds the olmoearth_run (OlmoEarth Studio) inference configs.

### Versions

#### 20260917_rslearn_bpcat_notemporal

- Source configs in rslearn_projects (`data/olmoearth_lcc/lcc_model/`):
  - Training: `config_rslearn_bpcat_notemporal.yaml` (rslearn_projects run
    `2026_09_17_lcc/rslearn_bpcat_notemporal`, trained on
    `/weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_dataset_20260811/`).
  - `model.yaml` is derived from `config_rslearn_bpcat_notemporal_predict.yaml`, the
    rslearn-only prediction config (no `rslp` imports). The model block must stay in
    sync with that file so the checkpoint loads strictly.
  - `dataset.json` is derived from `config_predict_rslearn.json` with the time offsets
    adjusted for Studio (see below) and the output layers reduced to one. The
    rslearn_projects configs pin Planetary Computer as the Sentinel-2 provider; here we
    use the olmoearth_run default (AWS COGs first, other providers as fallback). AWS
    COGs are pre-harmonized so the pixel values match the harmonized training imagery.
- Checkpoint:
  - WEKA: `/weka/dfive-default/rslearn-eai/projects/2026_09_17_lcc/rslearn_bpcat_notemporal/best.ckpt`
  - GCS copy for the platform: `gs://ai2-rslearn-projects-data/projects/2026_09_17_lcc/rslearn_bpcat_notemporal/best.ckpt`
- Requires rslearn >= 0.1.15 (for `BreakpointScan`, `TokensToChannels`,
  `Concatenate(concatenate_dim=TIME)`, and `OlmoEarth(token_pooling=false)`), so the
  Studio container image must use an olmoearth_run release that pins at least that
  version.

### Prediction Request Geometry

The model predicts change as of a reference date R. Set `oe_start_time` on each
feature to R; `oe_end_time` is ignored (set it equal to `oe_start_time`). The two image
layers derive their search ranges from R via `time_offset`/`duration` in
`dataset.json`:

- `sentinel2_frequent_0`: [R-60d, R], split into four 15-day periods with one
  least-cloudy mosaic each. Changes in the 15 days before R fall in the last period.
- `sentinel2_quarterly`: [R-1860d, R-60d], split into sixteen 90-day periods with one
  least-cloudy mosaic each.

Both layers use `min_matches` equal to `max_matches` (4 and 16), so a window fails
materialization if any period has no Sentinel-2 coverage. In practice this means R must
be at least ~5.1 years after the start of Sentinel-2 L2A availability for the region
(roughly 2017 onwards), and R-60d must be far enough in the past for the biweekly
imagery to be available.

Windows are 2048x2048 pixels at 10 m in the local UTM zone; the request geometry is
partitioned into 1x1 degree cells first.

### Output

Studio currently supports a single output per model, so only the `post_change` head is
written: a single uint8 band with the argmax fine-grained "what appeared" change
category. The class layout is

| value | label |
|-------|-------|
| 0 | nodata |
| 1 | none (no post-change category) |
| 2 | vegetation_growth |
| 3 | new_building |
| 4 | new_road |
| 5 | new_infrastructure |
| 6 | new_crop_field |
| 7 | new_aquafarm |
| 8 | site_clearing |
| 9 | water_expand |
| 10 | mining |
| 11 | new_crop_structure |
| 12 | selective_logging |
| 13 | landslide |
| 14 | settlement |

The model also predicts a binary change score, source and destination land cover
categories, a "what was removed" (`pre_change`) category, and the change start/end
timestep (`ts_start`/`ts_end`). These heads are still present in `model.yaml` (they
are needed for the checkpoint to load) but are not written; they can be exposed once
Studio supports multiple outputs (add an `RslearnWriter` per head and a matching
output layer, as in `config_rslearn_bpcat_notemporal_predict.yaml`).

### Running Locally

Imagery is fetched from OlmoEarth Datasets, so the credentials must be set:

```bash
export OEDATASETS_API_URL=https://datasets.olmoearth.allenai.org
export DATASETS_API_TOKEN=<your-token>
export NUM_WORKERS=32
export WANDB_PROJECT=olmoearth_lcc
export WANDB_NAME=olmoearth_lcc_inference
export WANDB_ENTITY=<your-entity>
```

Then, after editing `prediction_request_geometry.geojson` for the region and reference
date of interest:

```bash
python -m olmoearth_projects.main olmoearth_run olmoearth_run \
    --config_path $PWD/olmoearth_run_data/olmoearth_lcc/20260917_rslearn_bpcat_notemporal/ \
    --checkpoint_path gs://ai2-rslearn-projects-data/projects/2026_09_17_lcc/rslearn_bpcat_notemporal/best.ckpt \
    --scratch_path project_data/olmoearth_lcc/
```

The combined raster is written to `project_data/olmoearth_lcc/results/results_raster/`.

### Registering in Studio

In the olmoearth_run admin UI, create a model pipeline with one stage using the legacy
config format: paste `dataset.json`, `model.yaml`, and `olmoearth_run.yaml` from this
directory, set the checkpoint file path to the GCS checkpoint above, and pick a
container image whose olmoearth_run/rslearn versions satisfy the requirement above.
