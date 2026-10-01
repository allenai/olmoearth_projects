Forest Loss Driver: Monoculture Classifier
==========================================

This is the OlmoEarth Studio model package for the monoculture (monocrop) classifier
that runs as a second stage in the forest loss driver pipeline (see
`olmoearth_projects/projects/forest_loss_driver/deploy/`). Given a forest loss event
that the forest loss driver model classified as agriculture, it predicts which type of
agriculture the land was converted to.

Classes: `nodata` (never predicted in practice), `mennonites_nonsoybean`, `soybean`,
`oil_palm`, `other_agriculture`, `pastures`, `rice`.

Inputs
------

The model inputs 12 monthly Sentinel-2 L2A mosaics on a 128x128 window (UTM, 10
m/pixel) centered on the forest loss event. The window time range is taken from the
`oe_start_time` / `oe_end_time` properties of each request feature and must span
exactly 360 days (12 x 30-day mosaic periods).

The model is trained with `12 - m` pre-loss months followed by `m` post-loss months,
where `m` ranges from 1 to 12. Since the pipeline only runs the model on events from
the last year, it submits every event centroid with the same time range, the 360 days
ending at the run time:

```
oe_start_time = run_time - 360 days
oe_end_time   = run_time
```

Events less than 30 days old are not submitted since there is no post-loss imagery yet.
The 30-day mosaics are built from all available Sentinel-2 scenes, so windows rarely
fail, but the pipeline still uses a lower `min_window_success_rate` to tolerate cloudy
periods where a mosaic is missing (`min_matches` is 12).

Output
------

The model produces one vector feature per window with a `class_name` property (and
`probs`). The pipeline matches these back to the forest loss events by centroid.

Local validation
----------------

The model can be run locally with `olmoearth_run` on a small GeoJSON of points with
the shifted time ranges described above:

```
python -m olmoearth_projects.main olmoearth_run olmoearth_run \
  --config_path $PWD/olmoearth_run_data/forest_loss_driver_monocrop/ \
  --checkpoint_path /path/to/checkpoint.ckpt \
  --scratch_path /path/to/scratch/
```

History
-------

- 2026-09-15: Initial version using the window classification model
  `20260914_monocrop_classifier/olmoearth_v1_2_base_classify_pool` from
  rslearn_projects (`data/forest_loss_driver/monocrop_classifier/model_classify_pool.yaml`).
