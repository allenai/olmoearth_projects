Local olmoearth_run inference for Nandi:
```bash
export EXTRA_FILES_PATH=/weka/dfive-default/helios/checkpoints
export DATASET_PATH=/weka/dfive-default/yawenz/datasets/scratch_v5/dataset_0
export NUM_WORKERS=32
export TRAINER_DATA_PATH=/weka/dfive-default/yawenz/test/nandi
export WANDB_PROJECT=2025_10_03_nandi_crop_type
export WANDB_NAME=nandi_crop_type_segment_helios_base_S2_S1_ts_ws4_ps1_bs8_add_annotations_2
export WANDB_ENTITY=eai-ai2
python -m olmoearth_projects.main olmoearth_run olmoearth_run --config_path olmoearth_run_data/nandi/inference --scratch_path /weka/dfive-default/yawenz/datasets/scratch_v5/ --checkpoint_path /weka/dfive-default/yawenz/test/checkpoints/last_rewritten.ckpt
```
Note that the original `task_name` has been converted from `crop_type_classification` to `class` in the checkpoint, and this checkpoint is also available at `gs://earth-system-run-dev/models/4edd1efb-b645-44c3-8d7a-5cc2abbbcc46/stage_0/checkpoint.ckpt`.
