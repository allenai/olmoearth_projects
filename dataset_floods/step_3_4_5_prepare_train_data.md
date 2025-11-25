### Step 3: Prepare (fetch STAC items)

Queries Planetary Computer for Sentinel-1 and Sentinel-2 imagery matching your time windows.

```bash
rslearn dataset prepare --root ./dataset_floods --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5

```


### Step 4: Ingest

Ingests local label files into the rslearn dataset structure.

```bash
rslearn dataset ingest --root ./dataset_floods --workers 32

```

### Step 5: Materialize (final raster stacks)
Creates final georeferenced raster files for all layers.

```bash
rslearn dataset materialize --root ./dataset_floods --workers 32 \
  --retry-max-attempts 5 --retry-backoff-seconds 5 --ignore-errors

```

Note: make sure to have dataset_floods/config.json


```bash
{
  "layers": {
    "label": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["B1"],
          "dtype": "uint8",
          "nodata_value": -1
        }
      ],
    "data_source": {
      "name": "rslearn.data_sources.local_files.LocalFiles",
      "src_dir": "/home/wajahat/olmoearth_projects/sen1floods11_v1.5/data/LabelHand",
      "path_template": "{name}_LabelHand.tif",
      "ingest": true 
    },
      "resampling_method": "nearest"
    },
    "output": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["output"],
          "dtype": "uint8"
        }
      ]
    },
"sentinel2_l2a": {
  "type": "raster",
  "band_sets": [
    {
      "bands": [
        "B01","B02","B03","B04","B05","B06","B07",
        "B08","B8A","B09","B11","B12"
      ],
      "dtype": "uint16"
    }
  ],
  "data_source": {
    "name": "rslearn.data_sources.planetary_computer.Sentinel2",
    "cache_dir": "cache/planetary_computer",
    "ingest": false,
    "duration": "5d",
    "max_cloud_cover": 100,
    "query_config": {
      "period_duration": "5d",
      "max_matches": 1
    }
  }
}
,

    "sentinel1": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["vv", "vh"],
          "dtype": "float32"
        }
      ],
      "data_source": {
        "name": "rslearn.data_sources.planetary_computer.Sentinel1",
        "cache_dir": "cache/planetary_computer",
        "ingest": false,
        "duration": "5d",
        "query_config": {
          "period_duration": "5d",
          "max_matches": 1
        }
      }
    }
  }
}


```