"""Build the LCC model's unified OlmoEarthConfig.

The heads' decoder layers come from Studio's legacy model.yaml, which the spike showed
load Favyen's checkpoint strictly. Labels come from the HF training config, colors from
olmoearth_lcc_viewer's app/viewer/categories.py, and the Sentinel-2 windows from
Favyen's config_predict_rslearn.json.

Usage (from this directory): python make_olmoearth_config.py > olmoearth_config.yaml
"""

import re
import sys
from pathlib import Path

import yaml

# The recent-imagery window, which the quarterly window ends at. The rslearn prediction config
# (and training) uses 6 weekly mosaics from the last 90 days: 7-day periods can only span 84 or 91
# days, so 91. The legacy Studio config instead uses 4 biweekly mosaics over 60 days: 4, 15, 0, 60.
RECENT_PERIODS, RECENT_PERIOD_DAYS, RECENT_RECOVERY_PERIODS, RECENT_WINDOW_DAYS = (
    6,
    7,
    7,
    90,
)
NUM_TIMESTEPS = 16 + RECENT_PERIODS
# olmoearth_run fills unlabeled pixels of a segmentation label with 255 (SegmentationField.label_nodata).
SEGMENTATION_LABEL_NODATA = 255

LC_CLASSES = [
    "nodata",
    "bare",
    "burnt",
    "crops",
    "fallow/shifting cultivation",
    "grassland",
    "Lichen and moss",
    "shrub",
    "snow and ice",
    "tree",
    "urban/built-up",
    "water",
    "wetland (herbaceous)",
]
PRE_CLASSES = [
    "nodata",
    "none",
    "deforestation",
    "urban_erosion",
    "wetland_loss",
    "water_contract",
    "removed_crop_structure",
    "agricultural_activity",
    "wildfire",
    "ice_motion",
    "flooding",
]
POST_CLASSES = [
    "nodata",
    "none",
    "vegetation_growth",
    "new_building",
    "new_road",
    "new_infrastructure",
    "new_crop_field",
    "new_aquafarm",
    "site_clearing",
    "water_expand",
    "mining",
    "new_crop_structure",
    "selective_logging",
    "landslide",
    "settlement",
]
BINARY_CLASSES = ["nodata", "no_change", "change"]

# olmoearth_lcc_viewer app/viewer/categories.py. Classes it doesn't color (nodata, none) are transparent.
PRE_COLORS = {
    "deforestation": "#8b4513",
    "urban_erosion": "#708090",
    "wetland_loss": "#20b2aa",
    "water_contract": "#4682b4",
    "removed_crop_structure": "#daa520",
    # The viewer's "same change" categories, which share the pre head.
    "agricultural_activity": "#b8860b",
    "wildfire": "#e25822",
    "ice_motion": "#7fdbff",
    "flooding": "#001f3f",
}
POST_COLORS = {
    "vegetation_growth": "#2ecc40",
    "new_building": "#ff4136",
    "new_road": "#555555",
    "new_infrastructure": "#ff851b",
    "new_crop_field": "#9acd32",
    "new_aquafarm": "#00bcd4",
    "site_clearing": "#d2b48c",
    "water_expand": "#0074d9",
    "mining": "#b10dc9",
    "new_crop_structure": "#ffdc00",
    "selective_logging": "#3d9970",
    "landslide": "#85144b",
    "settlement": "#f012be",
}
LC_COLORS = {
    "bare": "#b4b4b4",
    "burnt": "#8b0000",
    "crops": "#f096ff",
    "fallow/shifting cultivation": "#d1a33d",
    "grassland": "#ffff4c",
    "Lichen and moss": "#fae6a0",
    "shrub": "#ffbb22",
    "snow and ice": "#f0f0f0",
    "tree": "#006400",
    "urban/built-up": "#fa0000",
    "water": "#0064c8",
    "wetland (herbaceous)": "#0096a0",
}
# The viewer shows change as a confidence ramp with #ff4136 above threshold; as an argmax band,
# only "change" is drawn.
BINARY_COLORS = {"change": "#ff4136"}

TRANSPARENT = [0, 0, 0, 0]


def rgba(hex_color: str) -> list[int]:
    """Convert #rrggbb to an opaque RGBA list."""
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [255]


def categorical(labels: list[str], colors: dict[str, str]) -> list[dict]:
    """Class values for labels, colored from colors and transparent otherwise."""
    return [
        {
            "value": i,
            "label": label,
            "color": rgba(colors[label]) if label in colors else TRANSPARENT,
        }
        for i, label in enumerate(labels)
    ]


def field(name: str, allowed_values: list[dict]) -> dict:
    """A segmentation output field."""
    return {
        "name": name,
        "field_type": "segmentation",
        "allowed_values": allowed_values,
    }


def date_field(name: str) -> dict:
    """A per-pixel date field: uint16 days since 1970-01-01, vectorized as ISO dates."""
    return {"name": name, "field_type": "per_pixel_date"}


# Band order follows Favyen's published summary raster where it can (argmax bands only for now).
FIELDS = [
    field("binary", categorical(BINARY_CLASSES, BINARY_COLORS)),
    field("pre_change", categorical(PRE_CLASSES, PRE_COLORS)),
    field("post_change", categorical(POST_CLASSES, POST_COLORS)),
    field("src", categorical(LC_CLASSES, LC_COLORS)),
    field("dst", categorical(LC_CLASSES, LC_COLORS)),
    date_field("ts_start"),
    date_field("ts_end"),
]
# How each labeled date maps to an input image in training (Favyen's training config).
# ts_start is the last date with no change yet, so it takes the image just before it,
# which can't show the change; ts_end is when the change is clear, so it takes the next image.
TIMESTEP_MODES = {"ts_start": "BEFORE", "ts_end": "AFTER"}


def timestep_task(
    name: str, task: dict, decoder_layers: list[dict]
) -> tuple[dict, list[dict]]:
    """The legacy timestep head as rslearn's per-pixel timestep task.

    The head still classifies each pixel into one of the input images, so the checkpoint
    loads unchanged, but the task writes the midpoint date of the chosen image (the
    mosaic's period) instead of its index.
    """
    assert decoder_layers[-1]["class_path"].endswith("SegmentationHead")
    head = {
        "class_path": "rslearn.train.tasks.per_pixel_timestep.PerPixelTimestepHead",
        # mode maps labeled dates to images, so it only matters for training.
        "init_args": {"input_key": "sentinel2_l2a", "mode": TIMESTEP_MODES[name]},
    }
    task = {
        "class_path": "rslearn.train.tasks.per_pixel_timestep.PerPixelTimestepTask",
        # Studio's legacy model.yaml says 20 timestep classes; the model was trained on 22 images.
        "init_args": {**task["init_args"], "num_classes": NUM_TIMESTEPS},
    }
    return task, [*decoder_layers[:-1], head]


def main(
    model_yaml: Path = Path(__file__).parent.parent
    / "20260917_rslearn_bpcat_notemporal"
    / "model.yaml",
) -> None:
    """Print the unified config built from the legacy model.yaml."""
    legacy = yaml.safe_load(re.sub(r"\$\{[A-Z_]+\}", "X", model_yaml.read_text()))
    mtm = legacy["model"]["init_args"]["model"]["init_args"]
    legacy_tasks = legacy["data"]["init_args"]["task"]["init_args"]["tasks"]
    encoder = mtm["encoder"][0]["init_args"]

    tasks = {}
    for f in FIELDS:
        task = legacy_tasks[f["name"]]
        decoder_layers = mtm["decoders"][f["name"]]
        if f["name"] in TIMESTEP_MODES:
            task, decoder_layers = timestep_task(f["name"], task, decoder_layers)
        else:
            num_classes = len(f["allowed_values"])
            task = {
                **task,
                "init_args": {
                    **task["init_args"],
                    "num_classes": num_classes,
                    "nodata_value": SEGMENTATION_LABEL_NODATA,
                },
            }
        tasks[f["name"]] = {
            "name": "manual",
            "decoder_layers": decoder_layers,
            "task": task,
        }

    config = {
        "config_version": "0.15.0",
        "data": {
            "temporality": {
                "name": "framed_point_in_time",
                "lookbehind_observations": [
                    # 16 quarterly mosaics from 20 periods (1800 days) ending where the recent window starts.
                    {
                        "offset_days": RECENT_WINDOW_DAYS,
                        "observation": {
                            "name": "repeating_interval",
                            "num_periods": 16,
                            "period_duration_days": 90,
                            "recovery_periods": 4,
                        },
                    },
                    # The recent mosaics, ending at the prediction date.
                    {
                        "offset_days": 0,
                        "observation": {
                            "name": "repeating_interval",
                            "num_periods": RECENT_PERIODS,
                            "period_duration_days": RECENT_PERIOD_DAYS,
                            "recovery_periods": RECENT_RECOVERY_PERIODS,
                        },
                    },
                ],
            },
            "modalities": {
                "sentinel2_l2a": {"space_mode": "mosaic", "sort_by": "cloud_cover"}
            },
            "output": {"data_type": "raster", "fields": FIELDS},
        },
        "model": {
            "encoder": {
                "name": "olmoearth",
                "patch_size": encoder["patch_size"],
                "output": "patch_tokens",
                "use_legacy_timestamps": encoder["use_legacy_timestamps"],
                "token_pooling": encoder["token_pooling"],
                "source": {"name": "huggingface", "model_id": encoder["model_id"]},
            },
            "tasks": tasks,
        },
        "input_preprocessing": {
            "default": {"input_size": 64, "input_mode": "all_tiles"},
            # Studio's defaults for fine-tuning a per-pixel model; olmoearth_run flips the date targets too.
            "train": {
                "input_mode": "single_random_crop",
                "additional_transforms": [
                    {"name": "random_flip", "x": True, "y": True}
                ],
            },
            "predict": {"overlap_pixels": 16},
        },
        # Studio's guided defaults for 64-pixel polygon labels; prediction ignores these two sections.
        "labeled_data_prep": {
            "geometry_preparation": {"line_string_policy": "centroid"},
            "sampler": {"name": "noop"},
            "window_preparer": {
                "name": "standard_raster",
                "windowing_mode": {
                    "name": "per_group",
                    "gridding_strategy": {"name": "pad_to_fill", "grid_size": 128},
                },
            },
            "data_splitter": {
                "name": "spatial",
                "grid_size": 512,
                "train_prop": 0.75,
                "val_prop": 0.25,
                "test_prop": 0.0,
            },
        },
        "training": {
            "max_epochs": 100,
            "batch_size": 8,
            "optimizer": {"name": "adamw", "lr": 0.0001},
            "scheduler": {
                "name": "plateau",
                "factor": 0.2,
                "patience": 2,
                "min_lr": 0.0,
                "cooldown": 10,
            },
            "encoder_training_strategy": {
                "name": "layer_wise_lr_decay",
                "decay_factor": 0.65,
                "num_layers": 12,
            },
            "early_stopping": {"patience": 15},
        },
        "prediction_requests": {
            "partitioners": {
                "request_to_partitions": {
                    "name": "grid",
                    "grid_size": 1.0,
                    "overlap_size": 0.0,
                    "projection": None,
                },
                "partition_to_windows": {
                    "name": "grid",
                    "grid_size": 2048,
                    "overlap_size": 0.0,
                    "projection": {"method": "use_utm"},
                },
            },
            "postprocessors": {
                "additional_steps": [
                    # 0 is nodata and 1 is "none", so only actual changes become polygons.
                    {
                        "name": "vectorize",
                        "field": "post_change",
                        "skip_label_ids": [0, 1],
                        # Percentiles rather than min/max so a few outlier pixels don't set the dates.
                        "auxiliary_fields": {
                            "binary": {"aggregation": "mode"},
                            "pre_change": {"aggregation": "mode"},
                            "src": {"aggregation": "mode"},
                            "dst": {"aggregation": "mode"},
                            "ts_start": {"aggregation": "percentile", "percentile": 10},
                            "ts_end": {"aggregation": "percentile", "percentile": 90},
                        },
                    }
                ]
            },
        },
        # Manual tasks name no metric to checkpoint on, so olmoearth_run needs one for training.
        "transpiler_context": {
            "downsample_factor": 1,
            "checkpoint_monitor": "val_loss",
            "checkpoint_mode": "min",
        },
    }
    yaml.safe_dump(config, sys.stdout, sort_keys=False, width=120)


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:]))
