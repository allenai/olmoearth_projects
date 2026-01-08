"""How well represented is each class in the labels?"""

import geopandas as gpd


def label_imbalance(df: gpd.GeoDataFrame) -> dict[str | int, float]:
    """How well represented is each class in the labels?

    For each class, values close to 1 / N are desirable, where
    N == the total number of classes
    """
    output_dict: dict[str | int, float] = {}
    for class_label in df.label.unique():
        output_dict[class_label] = len(df[df.label == class_label]) / len(df)
    return output_dict
