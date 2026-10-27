"""Download data for the few-shot segmentation example.

Reads annual OlmoEarth v1.3 embeddings over Ca Mau (Vietnam) from Source
Cooperative, averaged to 40 meters, then fetches a Sentinel-2 RGB composite
and ESA WorldCover aligned to the same grid.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.segmentation.download
"""

from __future__ import annotations

import argparse
from pathlib import Path

from olmoearth_embeddings_tutorial.common.constants import (
    EMBEDDINGS_FILENAME,
    S2_RGB_FILENAME,
    WORLDCOVER_FILENAME,
)
from olmoearth_embeddings_tutorial.common.imagery_sources import (
    download_s2_rgb,
    download_worldcover,
)
from olmoearth_embeddings_tutorial.common.source_coop import fetch_embeddings

CA_MAU_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [104.72, 8.53],
                        [104.72, 8.93],
                        [105.12, 8.93],
                        [105.12, 8.53],
                        [104.72, 8.53],
                    ]
                ],
            },
        }
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year of the annual embeddings. Default: %(default)s.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/segmentation"),
        help="Root directory for downloaded data.",
    )
    return parser.parse_args()


def main() -> None:
    """Download all data for Ca Mau."""
    args = parse_args()
    ca_mau_dir = args.out_dir / "ca_mau"

    print("Downloading Ca Mau embeddings...")
    ca_mau_embed = fetch_embeddings(
        CA_MAU_GEOJSON, args.year, ca_mau_dir / EMBEDDINGS_FILENAME
    )

    print("\nDownloading Sentinel-2 RGB...")
    download_s2_rgb(
        ca_mau_embed,
        ca_mau_dir / S2_RGB_FILENAME,
        datetime_range=f"{args.year}-01-01/{args.year}-12-31",
    )

    print("Downloading WorldCover...")
    download_worldcover(ca_mau_embed, ca_mau_dir / WORLDCOVER_FILENAME)

    print(f"\nDone. Data saved under: {ca_mau_dir}")


if __name__ == "__main__":
    main()
