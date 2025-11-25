import json
from pathlib import Path
from shapely import wkb

BASE_DIR = Path(
    "sen1floods11_data"
)
LABEL_DIR = BASE_DIR / "data" / "LabelHand"
SPLITS_DIR = BASE_DIR / "splits"


def build_windows(metadata_path: Path, out_path: Path):
    with metadata_path.open() as f:
        meta = json.load(f)

    windows = []
    for name, m in meta.items():
        bbox_hex = m.get("bbox_wkb")
        if not bbox_hex:
            raise RuntimeError(f"Missing bbox_wkb for chip {name}")

        try:
            bbox = wkb.loads(bytes.fromhex(bbox_hex)).bounds
        except Exception as exc:
            raise RuntimeError(f"Failed to decode bbox for {name}") from exc

        entry = {
            "name": name,
            "bbox": list(bbox),
            "crs": f"EPSG:{m.get('epsg', 4326)}",
            "datetime": m.get("timestamp"),
            "label_path": str(
                LABEL_DIR / f"{name}_LabelHand.tif"
            )
        }

        windows.append(entry)

    with out_path.open("w") as f:
        json.dump(windows, f, indent=2)


build_windows(
    SPLITS_DIR / "train_metadata.json",
    SPLITS_DIR / "windows_train.json"
)
build_windows(
    SPLITS_DIR / "val_metadata.json",
    SPLITS_DIR / "windows_val.json"
)
build_windows(
    SPLITS_DIR / "test_metadata.json",
    SPLITS_DIR / "windows_test.json"
)
