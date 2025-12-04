"""How much of the total labelled area does each class cover?"""

import geopandas as gpd
import numpy as np


def relative_area(bounds: tuple[float, float, float, float]) -> float:
    """Find the area of the bounding box.

    A = R^2 * (cos(f2)-cos(f1)) * (l2-l1) / 180
    Since we are using ratios, we omit the R and 180 constants.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    return np.cos(max_lat) - np.cos(min_lat) * (max_lon - min_lon)


def spatial_extent(df: gpd.GeoDataFrame) -> dict[str | int, float]:
    """How much of the total label spatial extent does each class occupy?

    For each class, values close to 1 are desirable.
    """
    total_area = relative_area(df.total_bounds)

    output_dict: dict[str | int, float] = {}
    for class_label in df.label.unique():
        label_area = relative_area(df[df.label == class_label].total_bounds)
        output_dict[class_label] = label_area / total_area

    return output_dict
