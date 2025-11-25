### Step 9: Prepare prediction items

```bash
rslearn dataset prepare --root ./dataset_floods --group predict --workers 32

```

### Step 10:  Materialize prediction rasters

```bash
rslearn dataset materialize --root ./dataset_floods --group predict --workers 32 \
  --retry-max-attempts 5 --retry-backoff-seconds 5 --ignore-errors

```

