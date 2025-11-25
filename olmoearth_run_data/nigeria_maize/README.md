# Nigeria Maize vs. not Maize classification

### Generating the data
```
python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/nigeria_maize/create_training_points.py --label_dir /weka/dfive-default/gabrielt/datasets/nigeria_maize

export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/nigeria_maize/20251125

python /weka/dfive-default/gabrielt/olmoearth_projects/olmoearth_projects/projects/nigeria_maize/create_windows.py --geojson_file /weka/dfive-default/gabrielt/datasets/nigeria_maize/combined.geojson --ds_path $DATASET_PATH --window_size 32
```
