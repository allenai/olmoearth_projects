## Transpiled rslearn configs (generated, for review)

What olmoearth_run turns `../olmoearth_config.yaml` into for a prediction. olmoearth_run generates these itself at run time; they're here so the unified config can be reviewed against the legacy files in `../../20260917_rslearn_bpcat_notemporal/`.

| File | Legacy counterpart |
|---|---|
| `dataset_config.json` | `dataset.json` |
| `model.yaml` | `model.yaml` |
| `inference_results_config.json` | the `inference_results_config` in `olmoearth_run.yaml` |

Generated with `transpile.py`, from olmoearth_run at `6dc58cfb` (run #643, pinning shared #211's `fc98a11`). `<dataset_path>` and `<model_stage_root>` stand for the per-prediction paths olmoearth_run fills in.

### What to check against the legacy configs

- **Model.** The seven decoder stacks match the legacy `model.yaml`. The encoder runs with `token_pooling: false` and `use_legacy_timestamps: false`. We loaded the checkpoint strictly into the model built from this file, and all seven heads' outputs were identical to the legacy model's on the same (synthetic) input.
- **Inputs.** There are two Sentinel-2 layers, fed to the model as one `sentinel2_l2a` input in this order:
  - `sentinel2_l2a_lookbehind_0`: 16 mosaics, 90-day periods, 1800 days ending 90 days before the reference date.
  - `sentinel2_l2a_lookbehind_1`: 6 mosaics, 7-day periods, the last 91 days.

  That's 22 images, as in `config_predict_rslearn.json` and training. The legacy Studio config uses 4 × 15-day mosaics (20 images). They join in time order like the legacy `Concatenate`, so the config needs no transform. `OlmoEarthNormalize` is added automatically, with the legacy band order.
- **Output.** The output is one `output` layer with a single 7-band uint8 band set: `binary`, `pre_change`, `post_change`, `src`, `dst`, `ts_start`, `ts_end`. `MultiFieldInstrumentedWriter` stacks each head's argmax in that order, and `inference_results_config.json` gives band N its labels and colors.
- **Prediction.** It uses 64-pixel crops with a 16-pixel overlap, as in the legacy config.

### Known differences, and open questions

- **Batch size is 32; legacy uses 8.** olmoearth_run takes the prediction batch size from the deployment (`PREDICTION_BATCH_SIZE`), not from the config. With 22 timesteps and unpooled tokens, 32 may not fit in GPU memory. We need either a deployment where it fits or a per-config override.
- **`autocast_dtype: float32`** appears only because these were transpiled on a CPU. olmoearth_run sets bfloat16 on a GPU.
- **The `labels` layer** is declared but prediction doesn't read it.
- **Timestep bands** hold an input-image index from 0 to 21. With nodata 0, a change at image 0 reads as "no prediction" until these bands are written as months, as in the published summary rasters.
- **Recent-imagery window: 22 images (6 weekly) or the legacy Studio 20 (4 biweekly)?** Set `RECENT_*` in `../make_olmoearth_config.py`, then rerun both scripts.

### Regenerating

After editing the unified config, run this from an olmoearth_run checkout with rslearn ≥ 0.1.16 installed:

```bash
PYTHONPATH=src:lib/shared/src python <path to this folder>/transpile.py
```
