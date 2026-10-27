"""Download embeddings + Sentinel-2 RGB for the PCA example.

Reads annual OlmoEarth v1.3 embeddings over Flevoland (Netherlands) from
Source Cooperative, averaged to 40 meters, and fetches a Sentinel-2 L2A median
composite aligned to the same grid.

Example:
    $ PYTHONPATH=src python -m olmoearth_embeddings_tutorial.pca.download
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

FLEVOLAND_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [5.35, 52.30],
                        [5.35, 52.60],
                        [5.80, 52.60],
                        [5.80, 52.30],
                        [5.35, 52.30],
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
        default=Path("data/flevoland"),
        help="Directory for downloaded data.",
    )
    return parser.parse_args()


def main() -> None:
    """Download embeddings and S2 RGB."""
    args = parse_args()

    print("Downloading Flevoland embeddings...")
    embed_path = fetch_embeddings(
        FLEVOLAND_GEOJSON, args.year, args.out_dir / EMBEDDINGS_FILENAME
    )

    print("Downloading Sentinel-2 RGB median composite...")
    s2_path = args.out_dir / S2_RGB_FILENAME
    download_s2_rgb(
        embed_path, s2_path, datetime_range=f"{args.year}-01-01/{args.year}-12-31"
    )

    print(f"Done. Embeddings: {embed_path}, S2 RGB: {s2_path}")


if __name__ == "__main__":
    main()
