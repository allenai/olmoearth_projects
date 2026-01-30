"""Fetches human-reviewed oil slick labels from the SkyTruth Cerulean API and prepares a GeoJSON for OlmoEarth.

This script downloads SkyTruth slick geometries that have undergone human-in-the-loop
(HITL) review, excludes BACKGROUND (1) and AMBIGUOUS (9) classifications as they are not true positives, and treats all
remaining classes (anthropogenic and natural) as positive oil slicks.

The output GeoJSON is intended to be consumed by `oer_annotation_creation.py` to generate
 annotation and task files. Optionally, the script can generate ring-based
negative examples around each slick geometry when explicit non-slick labels are not
available.

Example usage:

```bash
uv run python create_oil_spill_dataset.py \
    --out /Users/alexb/github/olmoearth_projects/olmoearth_run_data/oil_spills/labels.geojson \
    --limit 7000 \
    --make-ring-negatives
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry


CERULEAN_ENDPOINT = "https://api.cerulean.skytruth.org/collections/public.slick_plus/items"


def to_utc_iso(dt_like: Any) -> str:
    return pd.to_datetime(dt_like, utc=True).tz_convert("UTC").isoformat()


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def build_filter() -> str:
    # HITL present, exclude BACKGROUND (1) and AMBIGUOUS (9)
    # Use explicit EQ conditions matching Cerulean API syntax
    return "(hitl_cls EQ 2 OR hitl_cls EQ 3 OR hitl_cls EQ 4 OR hitl_cls EQ 5 OR hitl_cls EQ 6 OR hitl_cls EQ 7 OR hitl_cls EQ 8)"


def fetch_geojson(limit: int, timeout_s: float) -> Dict[str, Any]:
    params = {
        "limit": str(limit),
        "filter": build_filter(),
    }
    r = requests.get(CERULEAN_ENDPOINT, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def ring_negative(
    geom_wgs84: BaseGeometry,
    gap_m: float,
    width_m: float,
) -> Optional[BaseGeometry]:
    """
    ring = buffer(gap + width) - buffer(gap)
    Computed in EPSG:3857, returned in EPSG:4326.
    """
    if geom_wgs84 is None or geom_wgs84.is_empty:
        return None

    g3857 = gpd.GeoSeries([geom_wgs84], crs=4326).to_crs(3857).iloc[0]
    outer = g3857.buffer(gap_m + width_m, resolution=16)
    inner = g3857.buffer(gap_m, resolution=16)
    ring = outer.difference(inner)

    if ring.is_empty:
        return None

    return gpd.GeoSeries([ring], crs=3857).to_crs(4326).iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch Cerulean HITL slicks and prepare GeoJSON for oer_annotation_creation.py"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=7000)
    ap.add_argument("--timeout", type=float, default=60.0)

    ap.add_argument("--make-ring-negatives", action="store_true")
    ap.add_argument("--ring-gap-m", type=float, default=150.0)
    ap.add_argument("--ring-width-m", type=float, default=500.0)

    args = ap.parse_args()

    data = fetch_geojson(limit=args.limit, timeout_s=args.timeout)
    feats = data.get("features", [])
    print(f"Fetched {len(feats)} features from Cerulean API")
    if not feats:
        raise SystemExit("No features returned from Cerulean API.")

    gdf = gpd.GeoDataFrame.from_features(feats)
    gdf = ensure_wgs84(gdf)

    required = {"id", "hitl_cls", "slick_timestamp"}
    missing = required - set(gdf.columns)
    if missing:
        raise SystemExit(f"Missing expected fields from API: {missing}")

    out_features: List[Dict[str, Any]] = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        slick_id = str(row["id"])
        t = to_utc_iso(row["slick_timestamp"])

        # Positive slick
        out_features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "id": slick_id,
                    "start_time": t,
                    "end_time": t,
                    "label": "slick",
                    "hitl_cls": int(row["hitl_cls"]),
                },
            }
        )

        # Optional ring negative
        if args.make_ring_negatives:
            ring = ring_negative(geom, args.ring_gap_m, args.ring_width_m)
            if ring is not None and not ring.is_empty:
                out_features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(ring),
                        "properties": {
                            "id": f"{slick_id}_negative",
                            "start_time": t,
                            "end_time": t,
                            "label": "not_slick",
                            "derived": "ring_negative",
                            "source_slick_id": slick_id,
                            "ring_gap_m": args.ring_gap_m,
                            "ring_width_m": args.ring_width_m,
                        },
                    }
                )

    fc = {"type": "FeatureCollection", "features": out_features}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {args.out} with {len(out_features)} features "
        f"(ring_negatives={'on' if args.make_ring_negatives else 'off'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
