"""
Convert a GeoJSON of labeled polygons (e.g. Dhamar.geojson) into the two files that
`olmoearth_run` expects for fine-tuning:

- annotation_task_features.geojson
- annotation_features.geojson

`olmoearth_run` expects:
- Each task is a GeoJSON Feature with:
  - properties.oe_annotations_task_id (UUID)
  - properties.oe_start_time / properties.oe_end_time (ISO-8601 datetimes)
  - geometry: Polygon/MultiPolygon (task boundary)
- Each annotation is a GeoJSON Feature with:
  - properties.oe_annotations_task_id (UUID) (must match some task above)
  - properties.oe_labels: dict[str, int|float|None] (we use {"category": class_id})
  - optional oe_start_time / oe_end_time
  - geometry: Polygon/MultiPolygon (annotation geometry; we clip to the task boundary)


"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from shapely.geometry import mapping as shapely_mapping
from shapely.geometry import shape as shapely_shape
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from olmoearth_run.runner.models.training.annotation_features import AnnotationTaskFeature, AnnotationFeature, AnnotationTaskFeatureProperties, AnnotationFeatureProperties
from pydantic import BaseModel, Field, ConfigDict
from geojson_pydantic.features import FeatureCollection
from olmoearth_run.shared.models.model_stage_paths import (
    ANNOTATION_FEATURES_FILE_NAME,
    ANNOTATION_TASK_FEATURES_FILE_NAME)

# 9-class Yemen crop-type mapping.
# NOTE: These are 0-indexed class IDs (common for segmentation).
YEMEN_CROP_CLASSES: list[str] = [
    "orchards",
    "coffee",
    "inactive_cropland",
    "cereals",
    "not_cropland",
    "greenhouse",
    "fodder",
    "mixed_other",
    "qat",
]
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(YEMEN_CROP_CLASSES)}

LABEL_KEY = "crop_land"
# I want to do the very simple thing of creating the task_features file with the entire geometry from one geojson in a single task
class YemenCropLabelsProperties(BaseModel):
    start_time: datetime
    category: int | str

class YemenCropFeature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = Field(default="Feature")
    geometry: Polygon | MultiPolygon
    properties: YemenCropLabelsProperties

class YemenCropLabels(BaseModel):
    type: str = Field(default="FeatureCollection")
    features: list[YemenCropFeature]

# The start and end time are not the period when the label is valid but are more of a contextual choice that may eed to be configured and interjected


def dense_polygons_to_annotation_features(geojson_path: Path) -> tuple[dict, dict]:
    """
    Convert a GeoJSON of dense polygons into the two files that `olmoearth_run` expects for fine-tuning:
    - annotation_task_features.geojson
    - annotation_features.geojson
    """

    # Read the GeoJSON file
    with open(geojson_path, 'r') as f:
        labels = json.load(f)

    label_features = labels["features"]

    # ensure all the start and end times are the same
    # For now we will just use the first one
    oe_start_time = label_features[0]["properties"]["start_time"]
    oe_end_time = label_features[0]["properties"]["end_time"]

    all_features = [shapely_shape(feature["geometry"]) for feature in label_features]
    merged = unary_union(all_features)
    # Ensure result is a MultiPolygon for consistency
    merged_geometry = merged if isinstance(merged, MultiPolygon) else MultiPolygon([merged])
    task_id = uuid.uuid4()
    annotation_task_properties = AnnotationTaskFeatureProperties(
        oe_annotations_task_id=task_id,
        oe_start_time=oe_start_time,
        oe_end_time=oe_end_time,
    )
    annotation_task_feature = AnnotationTaskFeature(
        type="Feature",
        geometry=merged_geometry,
        properties=annotation_task_properties,
    )

    annotation_features = []
    for label_feature in label_features:
        label_geometry = shapely_shape(label_feature["geometry"])
        label_idx = CLASS_TO_ID[label_feature["properties"][LABEL_KEY]]

        annotation_properties = AnnotationFeatureProperties(
            oe_labels={LABEL_KEY: label_idx},
            oe_annotations_task_id=task_id,
            oe_start_time=oe_start_time,
            oe_end_time=oe_end_time,
        )

        annotation_feature = AnnotationFeature(
            type="Feature",
            geometry=label_geometry,
            properties=annotation_properties,
        )
        annotation_features.append(annotation_feature)
    return annotation_task_feature, annotation_features


def process_input(input_path: Path) -> tuple[list[AnnotationTaskFeature], list[AnnotationFeature]]:
    """
    Process either a single GeoJSON file or a directory of GeoJSON files.
    Returns combined lists of task features and annotation features.
    """
    all_task_features: list[AnnotationTaskFeature] = []
    all_annotation_features: list[AnnotationFeature] = []

    if input_path.is_file():
        geojson_files = [input_path]
    elif input_path.is_dir():
        geojson_files = sorted(input_path.glob("*.geojson"))
        if not geojson_files:
            raise ValueError(f"No .geojson files found in directory: {input_path}")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    for geojson_path in geojson_files:
        print(f"Processing: {geojson_path.name}")
        task_feature, annotation_features = dense_polygons_to_annotation_features(geojson_path)
        print("Features created!")
        all_task_features.append(task_feature)
        all_annotation_features.extend(annotation_features)

    print(f"Processed {len(geojson_files)} file(s): {len(all_task_features)} tasks, {len(all_annotation_features)} annotations")
    return all_task_features, all_annotation_features


def write_feature_collections(
    task_features: list[AnnotationTaskFeature],
    annotation_features: list[AnnotationFeature],
    output_dir: Path,
):
    """
    Write the task features and annotation features to GeoJSON FeatureCollection files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Write annotation features
    annotation_feature_collection = FeatureCollection(type="FeatureCollection", features=annotation_features)
    annotation_output_path = output_dir / ANNOTATION_FEATURES_FILE_NAME
    with open(annotation_output_path, 'w') as f:
        json.dump(annotation_feature_collection.model_dump(mode='json'), f)

    # Write task features
    task_feature_collection = {
        "type": "FeatureCollection",
        "features": [tf.model_dump(mode='json') for tf in task_features]
    }
    task_output_path = output_dir / ANNOTATION_TASK_FEATURES_FILE_NAME
    with open(task_output_path, 'w') as f:
        json.dump(task_feature_collection, f)

    print(f"Wrote {ANNOTATION_FEATURES_FILE_NAME} and {ANNOTATION_TASK_FEATURES_FILE_NAME} to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GeoJSON(s) of labeled polygons into the two files that `olmoearth_run` expects for fine-tuning")
    parser.add_argument("input", type=Path, help="Input GeoJSON file or directory containing .geojson files")
    parser.add_argument("output_dir", type=Path, help="The output directory for the annotation features")
    args = parser.parse_args()

    task_features, annotation_features = process_input(args.input)
    write_feature_collections(task_features, annotation_features, args.output_dir)