"""Download data for the change detection example.

Reads annual OlmoEarth v1.3 embeddings for the Park Fire region (Butte County,
California) from Source Cooperative for 2023 (before the fire) and 2025
(after), averaged to 40 meters. The fire burned from July to September 2024,
so neither year's composite mixes pre- and post-fire months. Also fetches a
September Sentinel-2 RGB composite for each year.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.change_detection.download
"""

from __future__ import annotations

import argparse
from pathlib import Path

from olmoearth_embeddings_tutorial.common.constants import (
    EMBEDDINGS_FILENAME,
    S2_RGB_FILENAME,
)
from olmoearth_embeddings_tutorial.common.imagery_sources import download_s2_rgb
from olmoearth_embeddings_tutorial.common.source_coop import fetch_embeddings

PARK_FIRE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-121.85, 39.70],
                        [-121.85, 39.85],
                        [-121.68, 39.85],
                        [-121.68, 39.70],
                        [-121.85, 39.70],
                    ]
                ],
            },
        }
    ],
}

BEFORE_YEAR = 2023
AFTER_YEAR = 2025


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/change_detection"),
        help="Root directory for downloaded data.",
    )
    return parser.parse_args()


def main() -> None:
    """Download embeddings and S2 RGB for both years."""
    args = parse_args()

    for year in (BEFORE_YEAR, AFTER_YEAR):
        year_dir = args.out_dir / str(year)
        print(f"\nDownloading {year} embeddings...")
        embed_path = fetch_embeddings(
            PARK_FIRE_GEOJSON, year, year_dir / EMBEDDINGS_FILENAME
        )
        print(f"Downloading Sentinel-2 RGB for September {year}...")
        download_s2_rgb(
            embed_path,
            year_dir / S2_RGB_FILENAME,
            datetime_range=f"{year}-09-01/{year}-10-01",
        )

    print(f"\nDone. Data saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
