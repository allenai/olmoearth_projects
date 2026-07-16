# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "streamlit>=1.39",
#     "streamlit-folium>=0.22",
#     "folium>=0.17",
#     "geopandas>=1.0",
#     "pandas>=2.0",
#     "requests>=2.31",
#     "shapely>=2.0",
# ]
# ///
"""Streamlit app to pull negative labels from the WorldCereal RDM.

Workflow:
    1. Define a bounding box (draw a rectangle on the map, or type coordinates).
    2. Search the RDM for all public datasets intersecting the bounding box.
    3. Fetch the points/polygons from the selected datasets.
    4. Select which WorldCereal legend classes count as negatives.
    5. Download the result as GeoJSON or CSV for use in OlmoEarth.

Run with:
    uv run scripts/rdm_negatives_app.py
"""

import io
import math
import sys

RDM_API = "https://ewoc-rdm-api.iiasa.ac.at"
LEGEND_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/"
    "legend/WorldCereal_LC_CT_legend_latest.csv"
)
# The items endpoint caps MaxResultCount at 2000.
PAGE_SIZE = 2000

# Datasets whose extent bbox has an IoU with the query bbox below this are
# unchecked by default in the dataset table (configurable in the UI). Global
# datasets intersect any query bbox, but with a tiny IoU.
DEFAULT_MIN_IOU = 0.01


def _bbox_area_km2(b: tuple[float, float, float, float]) -> float:
    """Approximate area (km2) of a lon/lat bounding box on a sphere."""
    minx, miny, maxx, maxy = (float(v) for v in b[:4])
    r_km = 6371.0
    return (
        abs(math.radians(maxx - minx))
        * abs(math.sin(math.radians(maxy)) - math.sin(math.radians(miny)))
        * r_km**2
    )


def bbox_area_mkm2(b: list | tuple | None) -> float | None:
    """Approximate area (in millions of km2) of a lon/lat bounding box."""
    if not b or len(b) < 4 or any(v is None for v in b[:4]):
        return None
    return round(_bbox_area_km2(tuple(b)) / 1e6, 2)


def bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union of two lon/lat bounding boxes (spherical areas)."""
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    if inter[0] >= inter[2] or inter[1] >= inter[3]:
        return 0.0
    inter_area = _bbox_area_km2(inter)
    union_area = _bbox_area_km2(a) + _bbox_area_km2(b) - inter_area
    return inter_area / union_area if union_area else 0.0


# LC_code prefixes from the legend. 10/11/12 are cropland classes,
# 16 is "no cropland", 17 is "no temporary crops" (may include permanent
# crops), and 20+ are non-cropland land cover classes.
CROPLAND_LC_CODES = {"10", "11", "12"}
TEMPORARY_CROP_LC_CODES = {"10", "11"}
UNKNOWN_LC_CODES = {"0"}


def ewoc_numeric_to_dashed(code: int) -> str:
    """Convert a numeric ewoc_code (e.g. 1101010000) to legend form (11-01-01-000-0)."""
    s = f"{int(code):010d}"
    return f"{s[0:2]}-{s[2:4]}-{s[4:6]}-{s[6:9]}-{s[9]}"


def main() -> None:
    """Run the Streamlit app."""
    import folium
    import geopandas as gpd
    import pandas as pd
    import requests
    import streamlit as st
    from folium.plugins import Draw
    from streamlit_folium import st_folium  # type: ignore[import-untyped]

    def as_points(geoms: gpd.GeoSeries) -> gpd.GeoSeries:
        """Return point geometries, taking centroids of polygons in a projected CRS."""
        if (geoms.geom_type == "Point").all():
            return geoms
        return geoms.to_crs("EPSG:6933").centroid.to_crs("EPSG:4326")

    st.set_page_config(page_title="RDM negatives", layout="wide")
    st.title("WorldCereal RDM negative sampler")
    st.markdown(
        "Pull reference data from the [WorldCereal RDM](https://rdm.esa-worldcereal.org) "
        "inside a bounding box, mark legend classes as negatives, and export a "
        "GeoJSON/CSV for OlmoEarth. Only **public** RDM collections are queried."
    )

    @st.cache_data(show_spinner="Downloading WorldCereal legend...")
    def fetch_legend() -> pd.DataFrame:
        resp = requests.get(LEGEND_URL, timeout=60)
        resp.raise_for_status()
        legend = pd.read_csv(
            io.BytesIO(resp.content), sep=";", dtype=str, encoding="utf-8-sig"
        )
        legend = legend.drop_duplicates(subset="ewoc_code", keep="first")
        legend["LC_code"] = legend["LC_code"].fillna("0")
        return legend.fillna("")

    @st.cache_data(show_spinner="Searching RDM collections...")
    def search_collections(bbox: tuple[float, float, float, float]) -> pd.DataFrame:
        params = [("Bbox", v) for v in bbox]
        resp = requests.get(f"{RDM_API}/collections/search", params=params, timeout=60)
        resp.raise_for_status()
        rows = []
        for c in resp.json():
            extent = c.get("extent", {})
            interval = extent.get("temporal", {}).get("interval", [[None, None]])
            spatial = extent.get("spatial", {}).get("bbox", [None])[0]
            rows.append(
                {
                    "collectionId": c["collectionId"],
                    "title": c.get("title", ""),
                    "featureCount": c.get("featureCount", 0),
                    "extent_Mkm2": bbox_area_mkm2(spatial),
                    "extent_bbox": tuple(spatial[:4])
                    if spatial and len(spatial) >= 4
                    else None,
                    "type": c.get("type", ""),
                    "accessType": c.get("accessType", ""),
                    "temporal": " to ".join(
                        str(t)[:10] for t in interval[0] if t is not None
                    ),
                }
            )
        return pd.DataFrame(rows)

    @st.cache_data(show_spinner=False)
    def fetch_items(
        collection_id: str, bbox: tuple[float, float, float, float], cap: int
    ) -> gpd.GeoDataFrame:
        params: list[tuple[str, float | int | str]] = [("Bbox", v) for v in bbox]
        features: list[dict] = []
        skip = 0
        while True:
            page_params = params + [
                ("MaxResultCount", min(PAGE_SIZE, cap - len(features))),
                ("SkipCount", skip),
            ]
            resp = requests.get(
                f"{RDM_API}/collections/{collection_id}/items",
                params=page_params,
                timeout=120,
                # the API redirects to an error page when a bbox query
                # matches zero items; treat that as an empty result
                allow_redirects=False,
            )
            if resp.is_redirect:
                break
            resp.raise_for_status()
            payload = resp.json()
            features.extend(payload.get("features", []))
            matched = payload.get("NumberMatched", 0)
            skip = len(features)
            if len(features) >= min(matched, cap) or not payload.get("features"):
                break
        if not features:
            return gpd.GeoDataFrame(
                columns=["sample_id", "ewoc_code", "valid_time", "geometry"],
                geometry="geometry",
                crs="EPSG:4326",
            )
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf["ref_id"] = collection_id
        return gdf

    # ------------------------------------------------------------------
    # Step 1: bounding box
    # ------------------------------------------------------------------
    st.header("1. Define a bounding box")
    map_col, coord_col = st.columns([2, 1])
    with map_col:
        fmap = folium.Map(location=[0.2, 37.5], zoom_start=5)
        Draw(
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "rectangle": True,
            },
            edit_options={"edit": False},
        ).add_to(fmap)
        map_state = st_folium(fmap, height=420, use_container_width=True)

    drawn_bbox = None
    drawing = (map_state or {}).get("last_active_drawing")
    if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
        coords = drawing["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        drawn_bbox = (min(lons), min(lats), max(lons), max(lats))

    with coord_col:
        st.markdown("Draw a rectangle on the map, or enter coordinates:")
        default = drawn_bbox or (36.0, -1.5, 38.0, 1.0)
        west = st.number_input("West (min lon)", value=float(default[0]), format="%.5f")
        south = st.number_input(
            "South (min lat)", value=float(default[1]), format="%.5f"
        )
        east = st.number_input("East (max lon)", value=float(default[2]), format="%.5f")
        north = st.number_input(
            "North (max lat)", value=float(default[3]), format="%.5f"
        )
        bbox = drawn_bbox or (west, south, east, north)
        source = "drawn rectangle" if drawn_bbox else "manual coordinates"
        st.caption(
            f"Using {source}: ({bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f})"
        )

    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        st.error("Invalid bounding box: min must be smaller than max.")
        st.stop()

    # ------------------------------------------------------------------
    # Step 2: find datasets
    # ------------------------------------------------------------------
    st.header("2. Datasets in the bounding box")
    if st.button("Search datasets", type="primary"):
        st.session_state["collections"] = search_collections(bbox)
        st.session_state["collections_bbox"] = bbox
        st.session_state.pop("items", None)
        st.session_state.pop("class_table", None)

    collections = st.session_state.get("collections")
    if collections is None:
        st.info("Draw/enter a bounding box, then click **Search datasets**.")
        st.stop()
    if st.session_state.get("collections_bbox") != bbox:
        st.warning(
            "The bounding box changed since the last search. "
            "Click **Search datasets** again to refresh."
        )

    if collections.empty:
        st.warning("No RDM collections intersect this bounding box.")
        st.stop()

    public = collections[collections.accessType == "Public"].copy()
    n_private = len(collections) - len(public)
    if n_private:
        st.caption(f"Skipping {n_private} non-public collection(s) (login required).")

    min_iou = st.number_input(
        "Minimum IoU between the query bounding box and a dataset's extent",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_MIN_IOU,
        step=0.005,
        format="%.3f",
        help="Datasets whose extent bounding box has an IoU with the query "
        "box below this are unchecked by default. Global datasets intersect "
        "any query box, but with a tiny IoU. Set to 0 to include everything.",
    )
    search_bbox = st.session_state["collections_bbox"]
    public["iou"] = public.extent_bbox.map(
        lambda b: bbox_iou(search_bbox, b) if b else 0.0
    )
    public = public.sort_values("iou", ascending=False)
    public.insert(0, "include", public.iou >= min_iou)
    n_below = int((~public.include).sum())
    if n_below:
        st.caption(
            f"{n_below} dataset(s) with IoU < {min_iou:g} are unchecked "
            "by default; tick them to include them anyway."
        )
    edited_collections = st.data_editor(
        public,
        hide_index=True,
        disabled=[c for c in public.columns if c != "include"],
        column_config={
            "include": st.column_config.CheckboxColumn("include"),
            "featureCount": st.column_config.NumberColumn("total features"),
            "iou": st.column_config.NumberColumn(
                "IoU",
                format="%.4f",
                help="Intersection-over-union between the query bounding box "
                "and the dataset's extent bounding box.",
            ),
            "extent_Mkm2": st.column_config.NumberColumn(
                "extent (M km2)",
                help="Approximate area of the dataset's bounding box, "
                "in millions of km2. Earth is ~510M km2.",
            ),
            "extent_bbox": None,
        },
        # changing the threshold resets the checkboxes to the new defaults
        key=f"collections_editor_{min_iou:g}",
    )
    selected_ids = edited_collections[edited_collections.include].collectionId.tolist()
    st.caption(f"{len(selected_ids)} collection(s) selected.")

    cap = int(
        st.number_input(
            "Max features to download per collection",
            min_value=PAGE_SIZE,
            value=20000,
            step=PAGE_SIZE,
            help="Collections with more matching features than this are truncated.",
        )
    )

    if st.button("Fetch points", type="primary", disabled=not selected_ids):
        frames = []
        progress = st.progress(0.0)
        for i, cid in enumerate(selected_ids):
            progress.progress(i / len(selected_ids), text=f"Fetching {cid}...")
            try:
                gdf = fetch_items(cid, bbox, cap)
            except requests.RequestException as e:
                st.warning(f"{cid}: failed to fetch ({e}); skipping.")
                continue
            if len(gdf) == cap:
                st.warning(f"{cid}: hit the {cap} feature cap; result truncated.")
            frames.append(gdf)
        progress.progress(1.0, text="Done")
        st.session_state["items"] = (
            pd.concat(frames, ignore_index=True)
            if frames
            else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        )
        st.session_state.pop("class_table", None)

    items: gpd.GeoDataFrame | None = st.session_state.get("items")
    if items is None or items.empty:
        if items is not None:
            st.warning("No features found in the selected collections.")
        st.stop()
        # st.stop() never returns; this return is for type checkers that
        # can't see that (e.g. mypy without streamlit installed)
        return

    per_collection = items.groupby("ref_id").size().rename("features fetched")
    st.dataframe(per_collection)

    # ------------------------------------------------------------------
    # Step 3: pick negative classes from the legend
    # ------------------------------------------------------------------
    st.header("3. Select negative classes")
    st.markdown(
        "These are the [WorldCereal legend]"
        "(https://artifactory.vgt.vito.be/ui/native/auxdata-public/worldcereal/legend/"
        "WorldCereal_LC_CT_legend_latest.pdf) classes present in the fetched data. "
        "Check the classes that should be treated as **negatives**."
    )
    legend = fetch_legend()

    if "class_table" not in st.session_state:
        counts = (
            items.assign(code=items.ewoc_code.map(ewoc_numeric_to_dashed))
            .groupby("code")
            .size()
            .rename("count")
            .reset_index()
        )
        table = counts.merge(legend, left_on="code", right_on="ewoc_code", how="left")
        table["label_full"] = table.label_full.fillna(table.code)
        table = table[
            [
                "code",
                "count",
                "label_full",
                "level_1",
                "level_2",
                "level_3",
                "sampling_label",
                "LC_code",
                "LC_label",
            ]
        ].fillna("")
        table.insert(0, "negative", False)
        st.session_state["class_table"] = table.sort_values(
            ["LC_code", "code"]
        ).reset_index(drop=True)
        st.session_state["class_table_version"] = 0

    def set_selection(mask: bool | pd.Series) -> None:
        """Set the "negative" column and refresh the class table editor."""
        st.session_state["class_table"]["negative"] = mask
        st.session_state["class_table_version"] += 1

    table = st.session_state["class_table"]
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Select non-cropland"):
        set_selection(~table.LC_code.isin(CROPLAND_LC_CODES | UNKNOWN_LC_CODES))
    if b2.button("Select non-temporary-crops"):
        set_selection(~table.LC_code.isin(TEMPORARY_CROP_LC_CODES | UNKNOWN_LC_CODES))
    if b3.button("Select all"):
        set_selection(True)
    if b4.button("Clear selection"):
        set_selection(False)

    edited_table = st.data_editor(
        st.session_state["class_table"],
        hide_index=True,
        disabled=[c for c in table.columns if c != "negative"],
        column_config={"negative": st.column_config.CheckboxColumn("negative")},
        key=f"class_editor_{st.session_state['class_table_version']}",
        height=420,
    )
    st.session_state["class_table"] = edited_table
    negative_codes = set(edited_table[edited_table.negative].code)
    st.caption(
        f"{len(negative_codes)} class(es) selected as negatives "
        f"({int(edited_table[edited_table.negative]['count'].sum())} features)."
    )
    if not negative_codes:
        st.stop()

    # ------------------------------------------------------------------
    # Step 4: export
    # ------------------------------------------------------------------
    st.header("4. Export")
    o1, o2, o3 = st.columns(3)
    category = o1.text_input("Category value", value="negative")
    max_per_class = int(
        o2.number_input("Max points per class (0 = all)", min_value=0, value=0)
    )
    seed = int(o3.number_input("Sampling seed", min_value=0, value=42))
    centroids = st.checkbox("Convert polygons to centroids", value=True)
    add_season = st.checkbox(
        "Add oe_start_time / oe_end_time from a season window", value=False
    )
    if add_season:
        s1, s2 = st.columns(2)
        season_start = s1.text_input("Season start (MM-DD)", value="03-01")
        season_end = s2.text_input("Season end (MM-DD)", value="08-30")

    out = items.copy()
    out["ewoc_code_str"] = out.ewoc_code.map(ewoc_numeric_to_dashed)
    out = out[out.ewoc_code_str.isin(negative_codes)]
    label_map = legend.set_index("ewoc_code")
    out["label"] = out.ewoc_code_str.map(label_map.label_full).fillna(out.ewoc_code_str)
    out["sampling_label"] = out.ewoc_code_str.map(label_map.sampling_label).fillna("")
    out["valid_time"] = pd.to_datetime(out.valid_time)
    out["year"] = out.valid_time.dt.year
    out["category"] = category

    if max_per_class:
        # groupby().head() returns a GeoDataFrame at runtime, but the
        # geopandas stubs type it as a plain DataFrame
        out = gpd.GeoDataFrame(
            out.sample(frac=1, random_state=seed)
            .groupby("ewoc_code_str")
            .head(max_per_class)
            .reset_index(drop=True)
        )
    if centroids:
        out["geometry"] = as_points(out.geometry)

    if add_season:
        try:
            out["oe_start_time"] = out.year.map(
                lambda y: f"{y}-{season_start}T00:00:00Z"
            )
            out["oe_end_time"] = out.year.map(lambda y: f"{y}-{season_end}T00:00:00Z")
            pd.to_datetime(out.oe_start_time)
            pd.to_datetime(out.oe_end_time)
        except (ValueError, TypeError):
            st.error("Season dates must be valid MM-DD values.")
            st.stop()

    keep = [
        "sample_id",
        "ref_id",
        "ewoc_code_str",
        "label",
        "sampling_label",
        "valid_time",
        "year",
        "category",
    ]
    if add_season:
        keep += ["oe_start_time", "oe_end_time"]
    out = gpd.GeoDataFrame(
        out[keep + ["geometry"]].rename(columns={"ewoc_code_str": "ewoc_code"}),
        geometry="geometry",
        crs="EPSG:4326",
    )
    out["valid_time"] = out.valid_time.dt.strftime("%Y-%m-%d")

    st.markdown(f"**{len(out)}** negative points ready for export.")
    preview = as_points(out.geometry)
    st.map(
        pd.DataFrame({"lat": preview.y, "lon": preview.x}).sample(
            n=min(len(out), 2000), random_state=seed
        ),
        size=10,
    )
    st.dataframe(pd.DataFrame(out.drop(columns="geometry")).head(50))

    csv_out = out.copy()
    csv_centroids = as_points(csv_out.geometry)
    csv_df = pd.DataFrame(csv_out.drop(columns="geometry"))
    csv_df["latitude"] = csv_centroids.y
    csv_df["longitude"] = csv_centroids.x

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download GeoJSON",
        data=out.to_json(),
        file_name="rdm_negatives.geojson",
        mime="application/geo+json",
        type="primary",
    )
    d2.download_button(
        "Download CSV",
        data=csv_df.to_csv(index=False),
        file_name="rdm_negatives.csv",
        mime="text/csv",
        type="primary",
    )


if __name__ == "__main__":
    try:
        from streamlit import runtime
    except ImportError:
        print(
            "Streamlit is not installed. Run with: uv run scripts/rdm_negatives_app.py"
        )
        sys.exit(1)
    if runtime.exists():
        main()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
        sys.exit(stcli.main())
