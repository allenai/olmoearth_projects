"""
Polygon to Raster Window Preparer.

This module provides a window preparer that converts polygon/multipolygon annotations
to raster labels. It creates a grid around the entire area and then splits the grid into windows.
ALL polygon labels intersected by the window are assigned to that window.
"""

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from typing import cast
import math
import numpy as np
from shapely.geometry import box
from olmoearth_run.runner.models.training.labeled_data import (
    AnnotationTask,
    LabeledWindow,
    RasterLabel,
)
from olmoearth_run.runner.tools.labeled_window_preparers.labeled_window_preparer import (
    RasterLabelsWindowPreparer,
)
from olmoearth_run.runner.tools.labeled_window_preparers.geometry_utils import (
    # compute_window_bounds,
    create_raster_labels_from_annotations,
    project_geometry_to_crs,
)
from olmoearth_run.runner.tools.labeled_window_preparers.rasterization_utils import (
    DEFAULT_NODATA_VALUE,
)
from rslearn.utils import STGeometry, get_utm_ups_crs


def grid_split_raster_labels(
    full_window_bounds: tuple[int, int, int, int],  # These are PIXEL coordinates
    full_raster_labels: list[RasterLabel],
    input_size: int,
    nodata_value: int,
    base_st_geometry: STGeometry,
    task_id: str,
) -> list[LabeledWindow]:
    # Extract pixel bounds
    minx_px, miny_px, maxx_px, maxy_px = full_window_bounds

    # Dimensions in pixels
    full_width_px = maxx_px - minx_px
    full_height_px = maxy_px - miny_px

    assert full_width_px % input_size == 0, f"Full width must be divisible by input size got {full_width_px} and {input_size}"
    assert full_height_px % input_size == 0, f"Full height must be divisible by input size got {full_height_px} and {input_size}"
    assert len(full_raster_labels) == 1, "Only a single raster of labels is currently supported"
    # Number of tiles (working in pixel space)
    num_tiles_x = math.ceil(full_width_px / input_size)
    num_tiles_y = math.ceil(full_height_px / input_size)

    labeled_windows = []

    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            # Tile bounds in PIXEL coordinates
            tile_minx_px = minx_px + (tile_x * input_size)
            tile_miny_px = miny_px + (tile_y * input_size)
            tile_maxx_px = tile_minx_px + input_size
            tile_maxy_px = tile_miny_px + input_size

            # Array slicing (also in pixels, but relative to array origin)
            # Array indices start at 0, so offset by the full bounds
            array_x_start = tile_x * input_size
            array_y_start = tile_y * input_size
            array_x_end = array_x_start + input_size
            array_y_end = array_y_start + input_size
            # Slice raster labels
            full_label = full_raster_labels[0]
            tile_array = full_label.value[array_y_start:array_y_end, array_x_start:array_x_end]
            if np.all(tile_array == nodata_value):
                continue
            tile_raster_labels = [RasterLabel(key=full_label.key, value=tile_array)]


            # Create tile geometry in PIXEL space with same projection
            tile_polygon = box(tile_minx_px, tile_miny_px, tile_maxx_px, tile_maxy_px)
            tile_st_geometry = STGeometry(
                base_st_geometry.projection,  # Same projection (includes resolution)
                tile_polygon,  # Polygon in pixel coordinates
                base_st_geometry.time_range,
            )

            tile_name = f"task_{task_id}_tile_{tile_y}_{tile_x}"
            labeled_windows.append(
                LabeledWindow(name=tile_name, st_geometry=tile_st_geometry, labels=tile_raster_labels)
            )

    return labeled_windows


def pad_raster_labels_to_input_size(
    full_raster_labels: list[RasterLabel],
    input_size: int,
    nodata_value: int = DEFAULT_NODATA_VALUE,
) -> list[RasterLabel]:
    """Pad raster labels to the next multiple of input_size."""
    padded_labels = []

    for label in full_raster_labels:
        height, width = label.value.shape

        # Calculate padded dimensions
        padded_height = math.ceil(height / input_size) * input_size
        padded_width = math.ceil(width / input_size) * input_size

        # Calculate padding amounts (pad on right and bottom)
        pad_height = padded_height - height
        pad_width = padded_width - width

        # Pad the array with nodata_value
        padded_array = np.pad(
            label.value,
            ((0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=nodata_value
        )

        padded_labels.append(RasterLabel(key=label.key, value=padded_array))

    return padded_labels


def pad_window_bounds_to_input_size(
    full_window_bounds: tuple[int, int, int, int],
    input_size: int,
) -> tuple[int, int, int, int]:
    """Pad window bounds to the next multiple of input_size."""
    minx_px, miny_px, maxx_px, maxy_px = full_window_bounds

    # Calculate current dimensions
    width = maxx_px - minx_px
    height = maxy_px - miny_px

    # Calculate padded dimensions
    padded_width = math.ceil(width / input_size) * input_size
    padded_height = math.ceil(height / input_size) * input_size

    # Extend max bounds (keep min bounds fixed)
    padded_maxx_px = minx_px + padded_width
    padded_maxy_px = miny_px + padded_height

    return (minx_px, miny_px, padded_maxx_px, padded_maxy_px)


def pad_st_geometry_to_input_size(
    base_st_geometry: STGeometry,
    full_window_bounds: tuple[int, int, int, int],
) -> STGeometry:
    """Update STGeometry to match padded window bounds."""
    minx, miny, maxx, maxy = full_window_bounds

    # Create new bounding box with padded dimensions
    padded_polygon = box(minx, miny, maxx, maxy)

    return STGeometry(
        base_st_geometry.projection,
        padded_polygon,
        base_st_geometry.time_range,
    )


def pad_raster_bounds_and_geometry(
        full_raster_labels: list[RasterLabel],
        base_st_geometry: STGeometry,
        full_window_bounds: tuple[int, int, int, int],
        input_size: int,
        nodata_value: int = DEFAULT_NODATA_VALUE,
) -> tuple[list[RasterLabel], STGeometry, tuple[int, int, int, int]]:
    """Pad raster labels, geometry, and bounds to the next multiple of input_size."""
    # Pad the full window bounds to the next multiple of the input size
    padded_window_bounds = pad_window_bounds_to_input_size(full_window_bounds, input_size)

    # Pad the full raster labels to the next multiple of the input size
    padded_raster_labels = pad_raster_labels_to_input_size(full_raster_labels, input_size, nodata_value)

    # Pad the base st geometry to the next multiple of the input size
    padded_st_geometry = pad_st_geometry_to_input_size(base_st_geometry, padded_window_bounds)

    return padded_raster_labels, padded_st_geometry, padded_window_bounds

# THIS Function doesn't assume the all positive nature of utm
def compute_window_bounds(window_geometry: STGeometry) -> tuple[int, int, int, int]:
    """Compute integer bounds for the window from the window geometry.

    Args:
        window_geometry: The window geometry

    Returns:
        Tuple of (minx, miny, maxx, maxy) in world coordinates
    """
    bounds = cast(BaseGeometry, window_geometry.shp).bounds


    minx = math.floor((bounds[0]))
    miny = math.floor((bounds[1]))
    maxx = math.ceil((bounds[2]))
    maxy = math.ceil((bounds[3]))

    return (minx, miny, maxx, maxy)


class DensePolygonToRasterWindowPreparer(RasterLabelsWindowPreparer):
    """
    Window preparer that converts dense polygon/multipolygon annotations to raster labels.

    This preparer creates gridded windows around the entire annotation task area,
    then splits the rasterized labels into smaller tiles.

    Key characteristics:
    - Multiple windows per task (tiled grid)
    - Each tile has aligned raster labels
    - Labels are uint8 raster arrays
    - Uses UTM projection for consistent resolution
    """

    def __init__(
        self,
        window_resolution: float = 10.0,
        input_size: int = 16,
        nodata_value: int = DEFAULT_NODATA_VALUE
    ):
        """
        Initialize the DensePolygonToRasterWindowPreparer.

        Args:
            window_resolution: Resolution in meters per pixel (default: 10.0)
            input_size: Size of each tile in pixels (default: 512x512)
            nodata_value: Value to use for nodata pixels
        """
        self.window_resolution = window_resolution
        self.input_size = input_size
        self.nodata_value = nodata_value


    def prepare_labeled_windows(
        self, annotation_task: AnnotationTask
    ) -> list[LabeledWindow[list[RasterLabel]]]:
        """
        Prepare labeled windows from polygon annotation tasks.

        This method creates one window per annotation task, using the task geometry
        as the window boundary. It rasterizes all polygon annotations within the task
        into a single uint8 raster label.

        Args:
            annotation_task: Single AnnotationTask object containing task context and annotations

        Returns:
            List containing one LabeledWindow object with raster labels, or empty list if no annotations
        """
        if not annotation_task.annotations:
            return []

        # Calculate CRS based on task centroid
        # First check what projection the task geometry is in
        print(f"Task geometry projection: {annotation_task.task_st_geometry.projection}")
        print(f"Task geometry projection CRS: {annotation_task.task_st_geometry.projection.crs}")

        task_centroid = cast(
            BaseGeometry, annotation_task.task_st_geometry.shp
        ).centroid
        task_bounds = cast(
            BaseGeometry, annotation_task.task_st_geometry.shp
        ).bounds
        print(f"Task bounds before projection: {task_bounds}")

        corner_coords = [
            (task_bounds[0], task_bounds[1]), # low left
            (task_bounds[2], task_bounds[3]), # high right
            (task_bounds[0], task_bounds[3]), # low right
            (task_bounds[2], task_bounds[1]), # high left
        ]
        print(f"Corner coords: {corner_coords}")
        corner_utm_crs = [get_utm_ups_crs(coord[0], coord[1]) for coord in corner_coords]
        print(f"Corner utm crs: {corner_utm_crs}")
        assert all(utm_crs == corner_utm_crs[0] for utm_crs in corner_utm_crs), "Corner utm crs are not all the same"
        utm_crs = corner_utm_crs[0]
        utm_crs = get_utm_ups_crs(task_centroid.x, task_centroid.y)
        # Extract the task geometry
        task_geom = annotation_task.task_st_geometry.shp
        if not isinstance(
            task_geom, (Polygon, BaseGeometry)
        ) or task_geom.geom_type not in ["Polygon", "MultiPolygon"]:
            raise ValueError(
                f"Expected Polygon or MultiPolygon for task, got {type(task_geom)} with geom_type {getattr(task_geom, 'geom_type', 'unknown')}"
            )

        # Convert to appropriate projection if needed
        projected_geometry = project_geometry_to_crs(
            task_geom, self.window_resolution, utm_crs
        )

        # Create the window geometry
        window_st_geometry = STGeometry(
            projected_geometry.projection,
            projected_geometry.shp,
            annotation_task.task_st_geometry.time_range,
        )

        # Create the full raster label by rasterizing all polygon annotations
        window_bounds = compute_window_bounds(projected_geometry)
        full_raster_labels = create_raster_labels_from_annotations(
            annotations=annotation_task.annotations,
            window_bounds=window_bounds,
            window_resolution=self.window_resolution,
            crs=utm_crs,
            nodata_value=self.nodata_value,
        )
        print(f"Full raster labels shape: {full_raster_labels[0].value.shape}")
        print(f"Window bounds before padding: {window_bounds}")

        # Pad to the next multiple of input_size
        padded_raster_labels, padded_st_geometry, padded_window_bounds = pad_raster_bounds_and_geometry(
            full_raster_labels=full_raster_labels,
            base_st_geometry=window_st_geometry,
            full_window_bounds=window_bounds,
            input_size=self.input_size,
            nodata_value=self.nodata_value,
        )

        print(f"Padded raster shape: {padded_raster_labels[0].value.shape}")
        print(f"Padded window bounds: {padded_window_bounds}")

        # Split the full window into tiles
        labeled_windows = grid_split_raster_labels(
            full_window_bounds=padded_window_bounds,
            full_raster_labels=padded_raster_labels,
            input_size=self.input_size,
            nodata_value=self.nodata_value,
            base_st_geometry=padded_st_geometry,
            task_id=str(annotation_task.task_id),
        )
        print(f"Created {len(labeled_windows)} labeled windows")


        return labeled_windows