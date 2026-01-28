#!/usr/bin/env python
"""Mosaic multiple GeoTIFF files into a single GeoTIFF."""

import argparse
from pathlib import Path

import rasterio
from rasterio.merge import merge


def mosaic_geotiffs(input_dir: str, output_path: str, pattern: str = "*.tif") -> None:
    """Mosaic all GeoTIFFs in a directory into a single file."""
    input_path = Path(input_dir)
    tif_files = sorted(input_path.glob(pattern))

    if not tif_files:
        raise ValueError(f"No files matching '{pattern}' found in {input_dir}")

    print(f"Found {len(tif_files)} files to mosaic")

    # Open all datasets
    datasets = [rasterio.open(f) for f in tif_files]

    try:
        # Merge
        mosaic, out_transform = merge(datasets)

        # Copy metadata from first file and update
        out_meta = datasets[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            }
        )

        # Write output
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        print(f"Mosaic saved to {output_path}")
    finally:
        for ds in datasets:
            ds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mosaic GeoTIFF files")
    parser.add_argument("input_dir", help="Directory containing GeoTIFF files")
    parser.add_argument("output", help="Output mosaic file path")
    parser.add_argument(
        "--pattern", default="*.tif", help="Glob pattern (default: *.tif)"
    )
    args = parser.parse_args()

    mosaic_geotiffs(args.input_dir, args.output, args.pattern)
