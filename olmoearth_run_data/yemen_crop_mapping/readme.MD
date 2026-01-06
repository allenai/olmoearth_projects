
Steps
Command to re run the windows if it was a mistake
`find /weka/dfive-default/henryh/helios/olmoearth_projects/yemen_dhamar_scratch_grid_filter/dataset/windows/spatial_split -name "items.json" -type f -delete`
1. Add dates to raw files ` python rslp/yemen_crop/add_correct_time_stamps.py --dir_path /weka/dfive-default/henryh/helios/rslearn_repos/rslearn_projects/yemen_crop_type/yemen_crop_labels --output_dir /weka/dfive-default/henryh/helios/rslearn_repos/rslearn_projects/yemen_crop_type/valid_time_yemen_crop_labels` This adds the dates witha default offset

Labels at ` /weka/dfive-default/henryh/helios/rslearn_repos/rslearn_projects/yemen_crop_type/valid_time_yemen_crop_labels/`
2. Prepare annotation features
3. Prepare labeled windows
4. Run build datset from windows