# Few-Shot Segmentation

Few-shot land-cover segmentation over Ca Mau, Vietnam using OlmoEarth
embeddings. Samples 60 labeled pixels (20 per class) from ESA WorldCover,
trains a logistic regression, and predicts every pixel in the region.

Three classes: mangrove, water, other.

## Usage

```bash
# 1. Download embeddings, imagery, and WorldCover
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.segmentation.download

# 2. Analyze and generate figure
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.segmentation.analyze \
    --data-dir data/segmentation/ca_mau \
    --out figures/
```

## Expected output

**Few-shot segmentation** (`figures/fewshot_60labels.png`)

Left: Sentinel-2 RGB with 60 labeled pixels (magenta dots). Center:
WorldCover reference. Right: predicted segmentation (weighted F1 ~ 0.77;
exact score depends on which pixels are sampled).

![few-shot segmentation](figures/fewshot_60labels.png)
