# OlmoEarth Embeddings Tutorial

End-to-end examples that compute, download, and analyze OlmoEarth embeddings
via the Studio API. These scripts reproduce the use cases from the
[custom embeddings blog post](https://blog.allenai.org/olmoearth-embeddings):

1. **Similarity search** -- cosine-similarity heatmap and patch mosaic over
   California's Central Valley.
2. **Per-pixel segmentation** -- few-shot land-cover segmentation from 60
   labeled pixels in Ca Mau (Vietnam).
3. **Change detection** -- wildfire burn scar detection via cosine distance
   between monthly embeddings (Park Fire, California, Sept 2023 vs Sept 2024).
4. **PCA exploration** -- false-color RGB from the first three principal
   components of Flevoland (Netherlands) embeddings.

Examples 1, 2, and 4 use **OlmoEarth-v1.3-Base** (128-dim embeddings)
at 40-meter resolution with 12 monthly periods (annual composites) of
Sentinel-2 L2A imagery. Example 3 uses the same encoder at 1 monthly period.

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

The compute scripts use the **OlmoEarth Studio API** to generate embeddings.
[Reach out](https://allenai.org/olmoearth) if you're interested in gaining
access. Once signed in, go to **Settings > API Keys** to create a key.
Set it as an environment variable or pass it via `--api-key`:

```bash
export OLMOEARTH_API_KEY="your-key-here"
```

Alternatively, you can compute embeddings locally using the open-source
OlmoEarth models without Studio access. See the
[rslearn embeddings guide](https://github.com/allenai/rslearn/blob/master/docs/examples/OlmoEarthEmbeddings.md)
for instructions. The analyze scripts work with any embedding COG regardless
of how it was produced.

## Quick start

Run all commands from this directory
(`tutorials/embeddings/`).

### 1. One-time setup

Create a Studio project and two embeddings models (annual and monthly):

```bash
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.setup_model
```

This writes `config.json` with `project_id`, `model_id` (annual), and
`monthly_model_id` that subsequent scripts read automatically. If you belong
to more than one Studio organization, pass `--organization-id` to choose where
the project is created.

### 2. Similarity search

```bash
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.similarity.compute --config config.json
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.similarity.analyze \
    --embed data/central_valley/embeddings.tif \
    --rgb data/central_valley/s2_rgb.tif \
    --out figures/
```

Outputs: `figures/similarity_heatmap.png`, `figures/similarity_mosaic.png`.

### 3. Few-shot segmentation

```bash
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.segmentation.compute --config config.json
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.segmentation.analyze \
    --data-dir data/segmentation/ca_mau \
    --out figures/
```

Output: `figures/fewshot_60labels.png`.

### 4. Change detection

```bash
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.compute --config config.json
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.analyze \
    --before-dir data/change_detection/sept_2023 \
    --after-dir data/change_detection/sept_2024 \
    --out figures/
```

Output: `figures/change_detection.png`.

### 5. PCA false-color

```bash
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.pca.compute --config config.json
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.pca.analyze \
    --embed data/flevoland/embeddings.tif \
    --rgb data/flevoland/s2_rgb.tif \
    --out figures/
```

Output: `figures/pca_false_color.png`.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for the analyze
steps. They import the same functions used by the CLI scripts, so there is
no duplicated logic. Run the corresponding `compute.py` first to download
data, then open a notebook to explore results interactively:

```bash
jupyter notebook notebooks/similarity.ipynb
```

## Directory layout

```
tutorials/embeddings/
    config.json                       # (generated) project/model IDs
    data/                             # (generated, git-ignored)
    figures/                          # (generated, git-ignored)
    notebooks/
        change_detection.ipynb        # interactive change detection
        pca.ipynb                     # interactive PCA analysis
        segmentation.ipynb            # interactive few-shot segmentation
        similarity.ipynb              # interactive similarity analysis
    src/
        olmoearth_embeddings_tutorial/
            setup_model.py            # one-time project + model creation
            common/
                embedding_utils.py    # load embedding COGs
                imagery_sources.py    # Sentinel-2 RGB + WorldCover download
                studio_client.py      # API client, polling, result download
            change_detection/
                analyze.py            # cosine distance + figure
                compute.py            # submit + download Park Fire (monthly)
            pca/
                analyze.py            # PCA false-color figure
                compute.py            # submit + download Flevoland
            segmentation/
                analyze.py            # few-shot linear probe + figure
                compute.py            # submit + download Ca Mau
            similarity/
                analyze.py            # heatmap + patch mosaic
                compute.py            # submit + download Central Valley
```

## Notes

- The **compute** scripts require network access and a valid API key. They
  submit predictions and download satellite imagery from Planetary Computer.
- The **analyze** scripts are standalone: they take local file paths and can
  run offline once the data is downloaded.
- Sentinel-2 median composites can take several minutes to build depending on
  scene count and area size.
