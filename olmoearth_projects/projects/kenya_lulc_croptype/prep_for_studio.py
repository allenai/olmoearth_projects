"""Prep a labels.geojson for studio."""

# hackey script to prepare labels.geojson as prepared by
# create_training_points.py so that its suitable for studio, as
# described by https://docs.olmoearth.allenai.org/add-dataset/#prepare-your-dataset

from datetime import datetime

import geopandas as gpd

# day, month. Short rain cultivation is from August to December.
START_MONTH, START_DAY = 8, 1  # August 1
END_MONTH, END_DAY = 12, 30  # dec 30

if __name__ == "__main__":
    labels = gpd.read_file("kenya_labels/labels.geojson")

    # we need a "maize_or_not" column, as we defined in the
    # studio project
    labels["maize_or_not"] = labels.apply(
        lambda x: "maize" if x.sampling_ewoc_code == "maize" else "not_maize", axis=1
    )
    labels["start_time"] = labels.apply(
        lambda x: datetime(x.year, START_MONTH, START_DAY).strftime("%Y-%m-%d"), axis=1
    )
    labels["end_time"] = labels.apply(
        lambda x: datetime(x.year, END_MONTH, END_DAY).strftime("%Y-%m-%d"), axis=1
    )

    labels.to_file("kenya_labels/labels_for_studio.geojson", driver="GeoJSON")
