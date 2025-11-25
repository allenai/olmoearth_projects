### Step 11: Run Predictions

Use the command below to generate predictions with your fine-tuned model:

```bash
rslearn model predict \
  --config dataset_floods/model.yaml \
  --ckpt_path checkpoints/epoch=92-step=2139.ckpt \
  --trainer.devices=1 \
  --data.predict_config.groups="[predict]"

```
Note: Update --ckpt_path to point to the exact checkpoint file you want to use.