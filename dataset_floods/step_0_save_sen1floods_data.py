"""
Convert full Sen1Floods11 dataset from S3 registry to TerraTorch format
with proper naming conventions and normalized chip IDs.

Updated to extract and save metadata (timestamps, coordinates) for OlmoEarth fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
# import rasterio
# from rasterio.transform import from_bounds

from hum_ai.data_engine.database.registry import get_registry
from hum_ai.data_engine.database.session import get_session
from hum_ai.data_engine.datasets import dataset, dataset_exists
from hum_ai.data_engine.ingredients import ObservationType

from shapely import wkb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def decode_bbox_wkb(bbox_wkb):
    """Convert WKB hex bbox → (minx, miny, maxx, maxy)."""
    if bbox_wkb is None:
        return None
    geom = wkb.loads(bytes.fromhex(bbox_wkb))
    return geom.bounds


def normalize_id(chip_id: str, target_type: str) -> str:
    """Normalize chip IDs across modalities."""
    if "LabelHand" in chip_id:
        return chip_id.replace("LabelHand", target_type)
    return chip_id


def get_base_chip_id(chip_id: str) -> str:
    """Remove modality-specific suffix."""
    for suffix in ["_LabelHand", "_S1Hand", "_S2Hand"]:
        if suffix in chip_id:
            return chip_id.replace(suffix, "")
    return chip_id


# =============================================================================
# FIXED FUNCTION: save_tiff() with band_names support
# =============================================================================
def save_tiff(data, path, bbox, epsg, nodata=0, band_names=None):
    """Save array as GeoTIFF with correct metadata and optional band names."""
    import rasterio
    import numpy as np
    from rasterio.transform import from_bounds

    if data.ndim == 2:
        data = data[np.newaxis, :]

    bands, h, w = data.shape
    minx, miny, maxx, maxy = bbox

    transform = from_bounds(minx, miny, maxx, maxy, w, h)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w,
        count=bands,
        dtype=data.dtype,
        transform=transform,
        crs=f"EPSG:{epsg}",
        nodata=nodata
    ) as dst:

        for i in range(bands):
            dst.write(data[i], i + 1)

            # NEW: set band names
            if band_names and i < len(band_names):
                dst.set_band_description(i + 1, band_names[i])


class Sen1Floods11Converter:
    """Converter for Sen1Floods11 dataset with metadata extraction."""

    def __init__(self, chip_size: int = 512):
        self.chip_size = chip_size
        self.s2_band_order = [
            ObservationType.SENTINEL2_COASTAL,
            ObservationType.SENTINEL2_BLUE,
            ObservationType.SENTINEL2_GREEN,
            ObservationType.SENTINEL2_RED,
            ObservationType.SENTINEL2_RED_EDGE1,
            ObservationType.SENTINEL2_RED_EDGE2,
            ObservationType.SENTINEL2_RED_EDGE3,
            ObservationType.SENTINEL2_NIR,
            ObservationType.SENTINEL2_RED_EDGE4,
            ObservationType.SENTINEL2_WATER_VAPOR,
            ObservationType.SENTINEL2_CIRRUS,
            ObservationType.SENTINEL2_SWIR1,
            ObservationType.SENTINEL2_SWIR2,
        ]

    def _get_dataset_name(self, split: str) -> str:
        return f"Sen1Floods11-hand-{self.chip_size}-{split}"

    def _to_dict(self, rows) -> Dict[str, np.ndarray]:
        out = {}
        for row in rows:
            if "chip" in row and isinstance(row["chip"], np.ndarray):
                out[row["location"]] = row["chip"]
        return out

    @staticmethod
    def _extract_chip_metadata(row: dict, split: str) -> dict:
        """Extract minimal metadata for OlmoEarth."""
        try:
            metadata = {
                "timestamp": row.get("time").isoformat() if row.get("time") else None,
                "centroid_lon": float(row.get("centroid_lon", 0.0)),
                "centroid_lat": float(row.get("centroid_lat", 0.0)),
                "epsg": int(row.get("epsg", 4326)),
                "split": split
            }

            if "bbox" in row and row["bbox"] is not None:
                metadata["bbox_wkb"] = row["bbox"].hex() if isinstance(row["bbox"], bytes) else None

            return metadata

        except Exception as e:
            logger.warning(f"Failed metadata extraction: {e}")
            return {
                "timestamp": None,
                "centroid_lon": 0.0,
                "centroid_lat": 0.0,
                "epsg": 4326,
                "split": split
            }

    def _save_metadata(self, chip_metadata: Dict[str, dict], splits_dir: Path, split: str):
        metadata_path = splits_dir / f"{split}_metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(chip_metadata, f, indent=2)
            logger.info(f"Saved metadata for {len(chip_metadata)} chips → {metadata_path}")
        except Exception as e:
            logger.error(f"Metadata save failed: {e}")

    def convert_split(self, split: str, out_dir: str, max_samples: Optional[int] = None) -> List[str]:
        logger.info(f"Processing split: {split}")

        dataset_name = self._get_dataset_name(split)
        out_root = Path(out_dir)
        s1_dir = out_root / "data" / "S1GRDHand"
        s2_dir = out_root / "data" / "S2L1CHand"
        label_dir = out_root / "data" / "LabelHand"
        splits_dir = out_root / "splits"

        for d in [s1_dir, s2_dir, label_dir, splits_dir]:
            d.mkdir(parents=True, exist_ok=True)

        with get_session(read_only=True) as session:
            registry = get_registry(session)

            if not dataset_exists(dataset_name, registry):
                logger.error(f"{dataset_name} not found")
                return []

            flood_dataset = dataset(
                dataset_name, registry,
                chip_type=ObservationType.SEN1FLOODS11_FLOOD_LABEL_HAND
            ).take_all()

            flood = self._to_dict(flood_dataset)
            s1_vv = self._to_dict(dataset(dataset_name, registry, chip_type=ObservationType.SENTINEL1_VV).take_all())
            s1_vh = self._to_dict(dataset(dataset_name, registry, chip_type=ObservationType.SENTINEL1_VH).take_all())

            s2_bands = {
                obs: self._to_dict(dataset(dataset_name, registry, chip_type=obs).take_all())
                for obs in self.s2_band_order
            }

            chip_ids = list(flood.keys())
            if max_samples is not None:
                chip_ids = chip_ids[:max_samples]

            processed = []
            chip_metadata = {}

            for idx, chip_id in enumerate(chip_ids):
                try:
                    base_chip_id = get_base_chip_id(chip_id)

                    # Extract metadata
                    for row in flood_dataset:
                        if row["location"] == chip_id:
                            chip_metadata[base_chip_id] = self._extract_chip_metadata(row, split)
                            meta = chip_metadata[base_chip_id]

                            bbox = decode_bbox_wkb(meta.get("bbox_wkb"))
                            epsg = meta.get("epsg", 4326)

                            if bbox is None:
                                bbox = (0, 0, self.chip_size, self.chip_size)
                            if epsg is None:
                                epsg = 4326
                            break

                    # LABEL — ADD BAND NAME
                    save_tiff(
                        flood[chip_id],
                        label_dir / f"{base_chip_id}_LabelHand.tif",
                        bbox=bbox,
                        epsg=epsg,
                        nodata=-1,
                        band_names=["label"]
                    )

                    # S1 — ADD BAND NAMES ["VV", "VH"]
                    s1_id = normalize_id(chip_id, "S1Hand")
                    if s1_id in s1_vv and s1_id in s1_vh:
                        s1_stack = np.stack([s1_vv[s1_id], s1_vh[s1_id]])
                        save_tiff(
                            s1_stack,
                            s1_dir / f"{base_chip_id}_S1Hand.tif",
                            bbox=bbox,
                            epsg=epsg,
                            nodata=0,
                            band_names=["VV", "VH"]
                        )

                    # S2 — ADD BAND NAMES FOR ALL 13 SENTINEL-2 BANDS
                    s2_id = normalize_id(chip_id, "S2Hand")
                    s2_stack = []
                    for obs in self.s2_band_order:
                        if s2_id in s2_bands[obs]:
                            s2_stack.append(s2_bands[obs][s2_id])
                        else:
                            s2_stack.append(np.zeros((self.chip_size, self.chip_size), dtype=np.float32))

                    s2_band_names = [
                        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
                        "B08", "B8A", "B09", "B10", "B11", "B12"
                    ]

                    save_tiff(
                        np.stack(s2_stack),
                        s2_dir / f"{base_chip_id}_S2Hand.tif",
                        bbox=bbox,
                        epsg=epsg,
                        nodata=0,
                        band_names=s2_band_names
                    )

                    processed.append(base_chip_id)

                except Exception as e:
                    logger.error(f"Failed to process chip {chip_id}: {e}")

            # Save split list
            split_map = {
                "train": "flood_train_data.txt",
                "val": "flood_valid_data.txt",
                "test": "flood_test_data.txt",
                "bolivia-test": "flood_bolivia_test_data.txt",
            }

            if split in split_map:
                split_file = splits_dir / split_map[split]
                with open(split_file, "w") as f:
                    for cid in processed:
                        f.write(f"{cid}\n")

            # Save metadata JSON
            self._save_metadata(chip_metadata, splits_dir, split)

            return processed

    def convert_dataset(self, out_dir: str, splits: List[str] = None, max_samples: Optional[int] = None):
        if splits is None:
            splits = ["train", "val", "test", "bolivia-test"]

        results = {}
        for split in splits:
            results[split] = self.convert_split(split, out_dir, max_samples)
        return results


def main():
    out_dir = "sen1floods11_data"
    max_samples = None

    logger.info(f"Starting Sen1Floods11 → {out_dir}")
    converter = Sen1Floods11Converter(chip_size=512)
    results = converter.convert_dataset(out_dir=out_dir, max_samples=max_samples)

    print("\nCONVERSION COMPLETED")
    print(json.dumps({k: len(v) for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
