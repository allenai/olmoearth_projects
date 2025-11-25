### Step 11: Run Predictions

Generates flood segmentation predictions on test windows using the trained model.
```bash
rslearn model predict \
  --config dataset_floods/model.yaml \
  --ckpt_path checkpoints/epoch=92-step=2139.ckpt \
  --trainer.devices=1 \
  --data.predict_config.groups="[predict]"
```
Note: Update --ckpt_path to point to the exact checkpoint file you want to use.