import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------------------------------------
# Set the path to your metrics CSV file
# Example: logs/lightning_logs/version_0/metrics.csv
# -------------------------------------------------------------
CSV_PATH = Path("logs/lightning_logs/version_0/metrics.csv")


# -------------------------------------------------------------
# Load metrics
# -------------------------------------------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Metrics file not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)


# -------------------------------------------------------------
# Basic summaries
# -------------------------------------------------------------
print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

miou_data = df.dropna(subset=["val_mean_iou"])
acc_data = df.dropna(subset=["val_accuracy"])
train_data = df.dropna(subset=["train_loss"])
val_data = df.dropna(subset=["val_loss"])

print(f"Total epochs trained: {int(miou_data['epoch'].max())}")

best_miou = miou_data["val_mean_iou"].max()
best_miou_epoch = int(miou_data.loc[miou_data["val_mean_iou"].idxmax(), "epoch"])
print(f"Best mIoU: {best_miou:.4f} (epoch {best_miou_epoch})")

best_acc = acc_data["val_accuracy"].max()
best_acc_epoch = int(acc_data.loc[acc_data["val_accuracy"].idxmax(), "epoch"])
print(f"Best Accuracy: {best_acc:.4f} (epoch {best_acc_epoch})")

print("\nFinal Epoch Stats:")
print(f"  Train Loss: {train_data['train_loss'].iloc[-1]:.4f}")
print(f"  Val Loss:   {val_data['val_loss'].iloc[-1]:.4f}")
print(f"  Val mIoU:   {miou_data['val_mean_iou'].iloc[-1]:.4f}")
print(f"  Val Acc:    {acc_data['val_accuracy'].iloc[-1]:.4f}")
print("=" * 60)


# -------------------------------------------------------------
# Plot metrics
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))


# Loss curves
axes[0].plot(train_data["epoch"], train_data["train_loss"], label="Train", linewidth=2)
axes[0].plot(val_data["epoch"], val_data["val_loss"], label="Val", linewidth=2)
axes[0].set_title("Training & Validation Loss", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)


# Accuracy
axes[1].plot(acc_data["epoch"], acc_data["val_accuracy"], linewidth=2)
axes[1].set_title("Validation Accuracy", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(alpha=0.3)


# mIoU
axes[2].plot(miou_data["epoch"], miou_data["val_mean_iou"], linewidth=2)
axes[2].axhline(
    y=best_miou,
    linestyle="--",
    alpha=0.5,
    label=f"Peak: {best_miou:.3f}"
)
axes[2].set_title("Validation Mean IoU", fontsize=12)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("mIoU")
axes[2].legend()
axes[2].grid(alpha=0.3)


# -------------------------------------------------------------
# Save + show figure
# -------------------------------------------------------------
version = CSV_PATH.parent.name  # e.g., "version_0"

output_file = f"training_metrics_{version}.png"

plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_file}")
plt.show()