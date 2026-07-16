"""Filter the global WorldCover/CGLS-LC100 reference labels to Kenya.

Source: "A global reference data set for land cover mapping at 10 m resolution"
(Lesiv et al., ESSD 2025), the IIASA/VITO/WU visual-interpretation reference
labels used to train CGLS-LC100 and ESA WorldCover. These are the training
*labels*, not the WorldCover output map.

    Zenodo:  https://zenodo.org/records/14871660
    File:    final_reference_data.csv  (~1.97 GB, 16.5M points @ 10 m)

Download the source CSV first, e.g.:
    curl -L -o final_reference_data.csv \
        "https://zenodo.org/records/14871660/files/final_reference_data.csv?download=1"

Source columns: validation_id, submission_item_id, sampleid, timestamp,
class_id, class_name, reference_year, center_x (lon), center_y (lat).

Output is a CSV of labelled points ready for OlmoEarth Studio (separate
latitude/longitude, a time range, and the class_name label), see
https://docs.olmoearth.allenai.org/model-fine-tuning
"""

import argparse

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.prepared import prep

# Reference year of this label set (aligned with Sentinel-2 imagery for 2015).
REF_YEAR = 2015
START_DATE = f"{REF_YEAR}-01-01"
END_DATE = f"{REF_YEAR}-12-31"


def main() -> None:
    """Filter the reference labels to one country and write a Studio-ready CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", required=True, help="final_reference_data.csv (from Zenodo 14871660)"
    )
    ap.add_argument("--countries", required=True, help="Natural Earth admin0 shapefile")
    ap.add_argument("--country", default="Kenya", help="ADMIN name to filter to")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    countries = gpd.read_file(args.countries)
    sel = countries[countries["ADMIN"] == args.country]
    assert len(sel) == 1, f"expected 1 {args.country} feature, got {len(sel)}"
    geom = sel.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    pgeom = prep(geom)  # fast repeated point-in-polygon tests
    print(
        f"{args.country} bbox: lon [{minx:.3f}, {maxx:.3f}], lat [{miny:.3f}, {maxy:.3f}]"
    )

    kept = []
    total = 0
    for chunk in pd.read_csv(args.src, chunksize=args.chunksize):
        total += len(chunk)
        # cheap bounding-box prefilter before the exact polygon test
        m = chunk["center_x"].between(minx, maxx) & chunk["center_y"].between(
            miny, maxy
        )
        sub = chunk[m]
        if len(sub):
            inside = sub[
                [
                    pgeom.contains(Point(x, y))
                    for x, y in zip(sub["center_x"], sub["center_y"])
                ]
            ]
            if len(inside):
                kept.append(inside)
        print(f"  scanned {total:,} rows, kept so far {sum(len(k) for k in kept):,}")

    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    print(f"\nTotal scanned: {total:,}; inside {args.country}: {len(df):,}")

    out = (
        pd.DataFrame(
            {
                "latitude": df["center_y"].round(6),
                "longitude": df["center_x"].round(6),
                "start_date": START_DATE,
                "end_date": END_DATE,
                "class_name": df["class_name"],
                "class_id": df["class_id"].astype("int64"),
                "reference_year": df["reference_year"].astype("int64"),
                "cluster_id": df["sampleid"].astype("int64"),
            }
        )
        .sort_values(["latitude", "longitude"])
        .reset_index(drop=True)
    )
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} points -> {args.out}")
    print(f"Unique 100 m clusters (sampleid): {out['cluster_id'].nunique():,}")
    print("\nClass distribution:")
    print(out["class_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
