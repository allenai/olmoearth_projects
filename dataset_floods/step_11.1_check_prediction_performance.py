"""
Evaluates model predictions against ground truth labels on test set.
Computes mIoU, F1 score, and accuracy metrics with spatial reprojection.
"""

import json
import numpy as np
from pathlib import Path
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.metrics import confusion_matrix, f1_score

with open("sen1floods11_data/splits/windows_test.json") as f:
    windows = json.load(f)

results = []

for w in windows:
    name = w["name"]
    label_path = Path(w["label_path"])
    pred_path = Path(f"dataset_floods/windows/predict/{name}/layers/output/output/geotiff.tif")
    
    if not pred_path.exists():
        print(f"⚠ Skipping {name}: no prediction found")
        continue
    
    if not label_path.exists():
        print(f"⚠ Skipping {name}: no label found")
        continue
    
    with rasterio.open(label_path) as label_src:
        label = label_src.read(1)
        label_meta = label_src.meta.copy()
    
    with rasterio.open(pred_path) as pred_src:
        pred_array = np.empty((label_meta['height'], label_meta['width']), dtype=pred_src.dtypes[0])
        
        reproject(
            source=rasterio.band(pred_src, 1),
            destination=pred_array,
            src_transform=pred_src.transform,
            src_crs=pred_src.crs,
            dst_transform=label_meta['transform'],
            dst_crs=label_meta['crs'],
            resampling=Resampling.nearest
        )
    
    pred = pred_array
    
    valid_mask = (label != 255) & (label != -1)
    pred_valid = pred[valid_mask]
    label_valid = label[valid_mask]
    
    tn, fp, fn, tp = confusion_matrix(label_valid, pred_valid, labels=[0, 1]).ravel()
    
    iou_class0 = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
    iou_class1 = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    miou = (iou_class0 + iou_class1) / 2
    
    f1 = f1_score(label_valid, pred_valid, average='binary', pos_label=1)
    if len(label_valid) == 0:
        acc = 0.0
    else:
        acc = (pred_valid == label_valid).mean()
    
    results.append({
        'name': name,
        'miou': miou,
        'iou_class0': iou_class0,
        'iou_class1': iou_class1,
        'f1': f1,
        'accuracy': acc
    })
    
    print(f"✓ {name}: mIoU={miou:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

mean_miou = np.mean([r['miou'] for r in results])
mean_f1 = np.mean([r['f1'] for r in results])
mean_acc = np.mean([r['accuracy'] for r in results])

print(f"\n{'='*60}")
print(f"TEST SET RESULTS ({len(results)} windows)")
print(f"{'='*60}")
print(f"Mean mIoU: {mean_miou:.4f}")
print(f"Mean F1: {mean_f1:.4f}")
print(f"Mean Accuracy: {mean_acc:.4f}")