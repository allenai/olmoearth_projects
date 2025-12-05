import geopandas as gpd
import pandas as pd

from olmoearth_projects.utils.label_quality.label_imbalance import label_imbalance


def test_spatial_clustering_classification() -> None:
    df = pd.DataFrame(
        {
            "City": ["Brasilia", "Santiago", "Bogota", "Caracas"],
            "Country": ["Brazil", "Chile", "Colombia", "Venezuela"],
            # highly clustered labels
            "label": ["South", "South", "North", "North"],
            "Latitude": [-15.78, -33.45, 4.60, 10.48],
            "Longitude": [-47.91, -70.66, -74.08, -66.86],
        }
    )
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    )
    output_dict = label_imbalance(gdf[["label", "geometry"]])
    for key in gdf.label.unique():
        assert output_dict[key] == 0.5
