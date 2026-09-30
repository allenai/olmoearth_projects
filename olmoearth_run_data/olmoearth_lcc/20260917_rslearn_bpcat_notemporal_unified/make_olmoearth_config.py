"""Build the LCC model's unified OlmoEarthConfig.

The heads' decoder layers come from Studio's legacy model.yaml, which the spike showed
load Favyen's checkpoint strictly. Labels come from the HF training config, colors from
olmoearth_lcc_viewer's app/viewer/categories.py, and the Sentinel-2 windows from
Favyen's config_predict_rslearn.json.

Usage (from this directory): python make_olmoearth_config.py > olmoearth_config.yaml
"""

import colorsys
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


def timesteps() -> list[dict]:
    """One class per input image, oldest first, on a blue-to-yellow ramp."""
    values = []
    for i in range(NUM_TIMESTEPS):
        r, g, b = colorsys.hsv_to_rgb(0.66 - 0.5 * i / (NUM_TIMESTEPS - 1), 0.8, 0.9)
        kind = f"quarterly {i + 1}" if i < 16 else f"recent {i - 15}"
        values.append(
            {
                "value": i,
                "label": f"image {i} ({kind})",
                "color": [int(r * 255), int(g * 255), int(b * 255), 255],
            }
        )
    return values


def field(name: str, allowed_values: list[dict]) -> dict:
    """A segmentation output field with nodata 0."""
    return {
        "name": name,
        "field_type": "segmentation",
        "nodata_value": 0,
        "allowed_values": allowed_values,
    }


# Band order follows Favyen's published summary raster where it can (argmax bands only for now).
FIELDS = [
    field("binary", categorical(BINARY_CLASSES, BINARY_COLORS)),
    field("pre_change", categorical(PRE_CLASSES, PRE_COLORS)),
    field("post_change", categorical(POST_CLASSES, POST_COLORS)),
    field("src", categorical(LC_CLASSES, LC_COLORS)),
    field("dst", categorical(LC_CLASSES, LC_COLORS)),
    field("ts_start", timesteps()),
    field("ts_end", timesteps()),
]


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
        num_classes = len(f["allowed_values"])
        # Studio's legacy model.yaml says 20 timestep classes; the model was trained on 22 images.
        task = {**task, "init_args": {**task["init_args"], "num_classes": num_classes}}
        tasks[f["name"]] = {
            "name": "manual",
            "decoder_layers": mtm["decoders"][f["name"]],
            "task": task,
        }

    config = {
        "config_version": "0.13.0",
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
            "predict": {"overlap_pixels": 16},
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
            "postprocessors": {"additional_steps": []},
        },
        "transpiler_context": {"downsample_factor": 1},
    }
    yaml.safe_dump(config, sys.stdout, sort_keys=False, width=120)


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:]))
