### Step 9: Prepare prediction items

Fetches STAC items for test windows from Planetary Computer.

```bash
rslearn dataset prepare --root ./dataset_floods --group predict --workers 32

```

### Step 10:  Materialize prediction rasters

Downloads and creates raster files for test set inference.

```bash
rslearn dataset materialize --root ./dataset_floods --group predict --workers 32 \
  --retry-max-attempts 5 --retry-backoff-seconds 5 --ignore-errors

```

