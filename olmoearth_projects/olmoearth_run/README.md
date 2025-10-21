Here is example:

```
python -m olmoearth_projects.main olmoearth_run olmoearth_run --config_path olmoearth_run_data/satlas/solar_farm/ --scratch_path /tmp/scratch/
```

So in `olmoearth_run/satlas/solar_farm/` we have:

- `dataset.json`: the rslearn dataset configuration file.
- `model.yaml`: the rslearn model configuration file.
- `olmoearth_run.yaml`: new YAML file containing oerun pre/post processing config.
- `prediction_request_geometry.geojson`: the GeoJSON input to the olmoearth_run partition and window generation.


In the `olmoearth_run_data/sample` directory, we can also run training window preparation, which
depends on:

- `dataset.json`: the rslearn dataset configuration file.
- `olmoearth_run.yaml`: new YAML file containiner the window_prep config
- `annotation_features.geojson`: annotation geojson FeatureCollection exported from Studio
- `annotation_task_features.geojson`: the Studio task geojson Features corresponding to the above

Run with:

```
uv run python -m olmoearth_projects.main olmoearth_run prepare_labeled_windows \
    --project_path $(pwd)/olmoearth_run_data/sample \
    --scratch_path /tmp/scratch
```

to produce a new dataset at:

```
/tmp/scratch/dataset
```
