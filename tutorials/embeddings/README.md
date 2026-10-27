# OlmoEarth Embeddings Tutorial

End-to-end examples that download and analyze OlmoEarth embeddings. These
scripts reproduce the use cases from the
[custom embeddings blog post](https://blog.allenai.org/olmoearth-embeddings):

1. **Similarity search** -- cosine-similarity heatmap and patch mosaic over
   California's Central Valley.
2. **Per-pixel segmentation** -- few-shot land-cover segmentation from 60
   labeled pixels in Ca Mau (Vietnam).
3. **Change detection** -- wildfire burn scar detection via cosine distance
   between annual embeddings (Park Fire, California, 2023 vs 2025).
4. **PCA exploration** -- false-color RGB from the first three principal
   components of Flevoland (Netherlands) embeddings.

All examples use the precomputed global
[OlmoEarth v1.3 embeddings](https://source.coop/olmoearth/olmoearth-v1-3-embeddings)
on Source Cooperative: annual 10-meter, 128-dimensional embeddings from
**OlmoEarth-v1.3-Base**, built from Sentinel-2, Sentinel-1, and Landsat
imagery. They are public, so no account or API key is needed. The download
scripts average the 10-meter vectors to 40 meters to keep files small. You can
also browse the embeddings in the
[Embeddings Explorer](https://olmoearth.allenai.org/embeddings).

If you need a different resolution, time period, or set of imagery sources,
you can compute custom embeddings in
[OlmoEarth Studio](https://olmoearth.allenai.org) (see
[Custom embeddings](#5-optional-custom-embeddings-with-olmoearth-studio)
below) or locally with the
[rslearn embeddings guide](https://github.com/allenai/rslearn/blob/master/docs/examples/OlmoEarthEmbeddings.md).
The analyze scripts work with any int8 embedding GeoTIFF that uses the same
quantization (see `common/embedding_utils.py`).

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

Run all commands from this directory
(`tutorials/embeddings/`).

### 1. Similarity search

```bash
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.similarity.download
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.similarity.analyze \
    --embed data/central_valley/embeddings.tif \
    --rgb data/central_valley/s2_rgb.tif \
    --out figures/
```

Outputs: `figures/similarity_heatmap.png`, `figures/similarity_mosaic.png`.

### 2. Few-shot segmentation

```bash
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.segmentation.download
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.segmentation.analyze \
    --data-dir data/segmentation/ca_mau \
    --out figures/
```

Output: `figures/fewshot_60labels.png`.

### 3. Change detection

```bash
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.download
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.analyze \
    --before-dir data/change_detection/2023 \
    --after-dir data/change_detection/2025 \
    --out figures/
```

Output: `figures/change_detection.png`.

### 4. PCA false-color

```bash
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.pca.download
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.pca.analyze \
    --embed data/flevoland/embeddings.tif \
    --rgb data/flevoland/s2_rgb.tif \
    --out figures/
```

Output: `figures/pca_false_color.png`.

### 5. (Optional) Custom embeddings with OlmoEarth Studio

Annual composites blur short events: the Park Fire burned from July to
September 2024, and by 2025 much of the scar had greened up again. With
OlmoEarth Studio you can compute embeddings for exactly the window you care
about. This step computes one-month embeddings for September 2023 and
September 2024 and reruns the change detection on them.

It requires a Studio account. [Reach out](https://allenai.org/olmoearth) if
you're interested in gaining access. Once signed in, go to
**Settings > API Keys** to create a key:

```bash
export OLMOEARTH_API_KEY="your-key-here"
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.custom
PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.analyze \
    --before-dir data/change_detection/custom/sept_2023 \
    --after-dir data/change_detection/custom/sept_2024 \
    --before-label "September 2023 (before)" \
    --after-label "September 2024 (after)" \
    --out figures/custom/
```

The script creates a project and a one-month OlmoEarth v1.3 embeddings model
(40 meters, Sentinel-2), then waits for both predictions, which usually takes
30 to 60 minutes. Pass `--organization-id` if you belong to more than one
organization, and `--project-id`/`--model-id` to reuse a model on later runs.

Output: `figures/custom/change_detection.png`.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for the analyze
steps. They import the same functions used by the CLI scripts, so there is
no duplicated logic. Run the corresponding `download.py` first, then open a
notebook to explore results interactively:

```bash
jupyter notebook notebooks/similarity.ipynb
```

## Directory layout

```
tutorials/embeddings/
    data/                             # (generated, git-ignored)
    figures/                          # (generated, git-ignored)
    notebooks/
        change_detection.ipynb        # interactive change detection
        pca.ipynb                     # interactive PCA analysis
        segmentation.ipynb            # interactive few-shot segmentation
        similarity.ipynb              # interactive similarity analysis
    src/
        olmoearth_embeddings_tutorial/
            common/
                embedding_utils.py    # load + (de)quantize embeddings
                imagery_sources.py    # Sentinel-2 RGB + WorldCover download
                rgb_stretch.py        # RGB display stretch
                source_coop.py        # read global embeddings from Source Coop
                studio_client.py      # Studio API client (custom embeddings)
            change_detection/
                analyze.py            # cosine distance + figure
                custom.py             # Studio monthly embeddings (optional)
                download.py           # Park Fire, 2023 and 2025
            pca/
                analyze.py            # PCA false-color figure
                download.py           # Flevoland
            segmentation/
                analyze.py            # few-shot linear probe + figure
                download.py           # Ca Mau
            similarity/
                analyze.py            # heatmap + patch mosaic
                download.py           # Central Valley
```

## Notes

- The **download** scripts need network access. They read embeddings from
  Source Cooperative and satellite imagery from Planetary Computer. Each takes
  `--year` (2021 to 2025 everywhere; California also has 2017 to 2020).
- Reading an area downloads its full 10-meter embeddings before averaging, so
  the larger areas (Central Valley, Flevoland) move a few GB and can take a
  few minutes.
- The **analyze** scripts are standalone: they take local file paths and can
  run offline once the data is downloaded.
- Sentinel-2 median composites can take several minutes to build depending on
  scene count and area size.
