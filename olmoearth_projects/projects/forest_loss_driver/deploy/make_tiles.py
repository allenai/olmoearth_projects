"""Produce the tiles that will appear in the web app."""

import json
import multiprocessing
import os
import subprocess  # nosec
import tempfile

import tqdm
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_projects.utils.fs import copy_file


def make_tiles(workers: int, in_fname: str, gcs_ds_root: str) -> None:
    """Produce the vector tiles that will appear in the web app.

    Args:
        workers: number of workers to use for copying tile images to GCS.
        in_fname: the input GeoJSON filename.
        gcs_ds_root: the directory on GCS to store the tiles.
    """
    dst_upath = UPath(gcs_ds_root)

    # Add a tippecanoe dict to the GeoJSON features that tells tippecanoe to put the
    # forest loss events for each driver category into a separate layer.
    # Here we also remove properties we don't need in the tiles. And rename
    # oe_start_time to date which is used for time filter in the web app.
    with UPath(in_fname).open() as f:
        fc = json.load(f)

    for feat in fc["features"]:
        # Add tippecanoe dict, it goes on the Feature itself, not in the properties.
        # We need to add this dict so that's why we don't use GeojsonVectorFormat.
        feat["tippecanoe"] = {
            "layer": feat["properties"]["category"],
        }
        # Web app expects the date property.
        feat["properties"]["date"] = feat["properties"]["oe_start_time"]
        for prop_name in [
            "pre_assets",
            "post_assets",
            "category",
            "probs",
            "oe_start_time",
            "oe_end_time",
            "country",
            "center_pixel",
            "tif_fname",
        ]:
            if prop_name not in feat["properties"]:
                continue
            del feat["properties"][prop_name]

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_fname = UPath(tmp_dir) / "tmp.geojson"
        with local_fname.open("w") as f:
            json.dump(fc, f)

        # Apply tippecanoe to convert the GeoJSON into a set of vector tiles.
        local_tile_dir = os.path.join(tmp_dir, "tiles")
        subprocess.call(
            [
                "tippecanoe",
                # Choose the maximum zoom level automatically.
                "-zg",
                # Write to directory.
                "-e",
                local_tile_dir,
                # Need no compression in the tile files to work with Leaflet.js.
                "--no-tile-compression",
                # Drop the smaller polygons in coarser zoom levels.
                "--drop-smallest-as-needed",
                local_fname,
            ]
        )  # nosec

        # Copy to GCS from which we serve the tiles.
        src_fnames = UPath(local_tile_dir).glob("*/*/*.pbf")
        copy_jobs = []
        for src_fname in src_fnames:
            dst_fname = (
                dst_upath
                / src_fname.parents[1].name
                / src_fname.parents[0].name
                / src_fname.name
            )
            copy_jobs.append(
                dict(
                    src_fname=src_fname,
                    dst_fname=dst_fname,
                )
            )

        p = multiprocessing.Pool(workers)
        outputs = star_imap_unordered(p, copy_file, copy_jobs)
        for _ in tqdm.tqdm(outputs, total=len(copy_jobs)):
            pass
        p.close()
