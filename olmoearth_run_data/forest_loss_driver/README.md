History
-------

- 2026-04-20: Replace model.yaml with
  20260401_forest_loss_driver_peru_phase2/config.yaml_airstripfix_02
  which is trained with new annotations in Peru that ACA completed in
  March 2026.
- 2026-11-22: Replace the model with
  20261120_forest_loss_driver_hierarchy/config.yaml_00 (best epoch 46) from
  rslearn_projects `data/forest_loss_driver/20261120_hierarchy/config.yaml`, trained on
  6,297 ACA annotations with the three-level label hierarchy. It predicts the 22 leaves
  of the hierarchy (`driver` property, with leaf probabilities in `probs`) instead of
  the 10 flat categories; the deploy pipeline derives the Level 1/2 nodes and the
  legacy 10-class `category` from the leaf probabilities (see
  `olmoearth_projects/projects/forest_loss_driver/deploy/hierarchy.py`). Val (622
  windows): set accuracy 0.635, leaf 0.583, Level 1 0.835, Level 2 0.715, legacy
  0.786.
  - Uses OlmoEarth-v1.2-Base and 128x128 windows in the local UTM zone at 10 m/pixel
    (center-cropped to 64x64), as in training. The Sentinel-2 layers in
    `dataset.json` are unchanged.
  - `model.yaml` only uses rslearn classes: the training head and task from
    rslearn_projects (`HierarchicalClassificationHead`, `HierarchicalForestLossTask`)
    only differ in the loss and label format, so `ClassificationHead` and
    `ClassificationTask` give the same predictions and the checkpoint loads strictly.
  - Checkpoint: `gs://ai2-rslearn-projects-data/projects/20261120_forest_loss_driver_hierarchy/config.yaml_00/best.ckpt`
    (WEKA: `/weka/dfive-default/rslearn-eai/projects/20261120_forest_loss_driver_hierarchy/config.yaml_00/best.ckpt`).
  - Studio model `7606e593-ba70-48cb-b5f9-f032771a155f` in the Forest Loss Driver
    Colombia 12 Demo project. This is a legacy (not unified) config: a unified config
    can describe the model, but Studio rejects unified-config requests whose
    observation window extends into the future, which the 180-day post-loss window of
    every recent event does.
  - The monoculture model is no longer run by the deploy pipeline; events from earlier
    runs keep their `monoculture_category`.
