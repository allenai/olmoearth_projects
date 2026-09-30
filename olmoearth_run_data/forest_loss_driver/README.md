History
-------

- 2026-09-25: Replace model.yaml with
  20260924_forest_loss_driver_utm/config.yaml_02 from rslearn_projects
  (`data/forest_loss_driver/config.yaml`). It uses OlmoEarth-v1.2-Base with
  layer-wise learning rate decay, and is trained on a dataset rebuilt directly
  from the Studio labels (adding the "Validatetest" labels across Brazil, Peru,
  Bolivia, Colombia, and Ecuador) with 128x128 windows at 10 m/pixel in UTM.
  olmoearth_run.yaml now creates UTM windows to match, and dataset.json is
  copied from `data/forest_loss_driver/config.json`. The deploy pipeline runs it
  as Studio model `d3db659a-fe9e-4749-9c14-d7088d18bbb8` in the "Forest Loss
  Driver Model" project (Amazon Conservation Association organization).

- 2026-04-20: Replace model.yaml with
  20260401_forest_loss_driver_peru_phase2/config.yaml_airstripfix_02
  which is trained with new annotations in Peru that ACA completed in
  March 2026.
