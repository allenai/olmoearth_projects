"""Create prediction GeoJSON for forest loss driver classification from GLAD alerts."""

import math
import multiprocessing
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import fiona
import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.features
import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
import tqdm
from rasterio.crs import CRS
from rslearn.const import SHAPEFILE_AUX_EXTENSIONS, WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.fsspec import get_upath_local, open_rasterio_upath_reader
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_raster_projection_and_bounds
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

from olmoearth_projects.utils.logging import get_logger

logger = get_logger(__name__)

# Time corresponding to 0 in alertDate GeoTIFF files.
BASE_DATETIME = datetime(2019, 1, 1, tzinfo=UTC)

# Create windows at WebMercator zoom 13 (512x512 tiles).
WEB_MERCATOR_CRS = CRS.from_epsg(3857)
WEB_MERCATOR_M = 2 * math.pi * 6378137
PIXEL_SIZE = WEB_MERCATOR_M / (2**13) / 512
WEB_MERCATOR_PROJECTION = Projection(WEB_MERCATOR_CRS, PIXEL_SIZE, -PIXEL_SIZE)

ANNOTATION_WEBSITE_MERCATOR_OFFSET = 512 * (2**12)


@dataclass
class ExtractAlertsArgs:
    """Arguments for extract_alerts_pipeline.

    Args:
        gcs_tiff_filenames: the list of GCS TIFF filenames to extract alerts from.
        out_fname: the filename to write the prediction request geometry.
        country_data_path: the path to the country shapefile. It should be downloaded
            and extracted from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
        countries: limit alerts to those falling in these countries. It is a list of
            two-letter uppercase country codes, e.g. ["PE"] for Peru only.
        conf_prefix: the prefix for the confidence raster of the forest loss alerts.
        date_prefix: the prefix for the date raster of the forest loss alerts.
        prediction_utc_time: the UTC time of the prediction. This defaults to the
            current timestamp, but could be set to the past to look for historical
            forest loss drivers.
        min_confidence: the minimum confidence threshold.
        days: the number of days to consider before the prediction time.
        min_area: the minimum area threshold for an event to be extracted.
        max_number_of_events: the maximum number of events to extract per GLAD tile.
        workers: number of parallel worker processes to use for extracting events.
    """

    gcs_tiff_filenames: list[str]
    out_fname: str

    country_data_path: str = "./ne_10m_admin_0_countries.shp"
    countries: list[str] | None = None

    conf_prefix: str = "gs://earthenginepartners-hansen/S2alert/alert/"
    date_prefix: str = "gs://earthenginepartners-hansen/S2alert/alertDate/"
    prediction_utc_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    min_confidence: int = 2
    days: int = 160
    min_area: float = 16.0
    max_number_of_events: int | None = None
    workers: int = 8


def load_country_polygons(
    country_data_path: UPath, countries: list[str]
) -> dict[str, shapely.Geometry]:
    """Get the polygons corresponding to the specified countries.

    country_data_path should point to the shapefile downloaded and extracted from
    https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/,
    and the parent directory must contain the other auxiliary files too.
    """
    logger.info(f"loading country polygon from {country_data_path}")
    prefix = ".".join(country_data_path.name.split(".")[:-1])
    aux_files: list[UPath] = []
    for ext in SHAPEFILE_AUX_EXTENSIONS:
        aux_files.append(country_data_path.parent / (prefix + ext))
    country_wgs84_shps: dict[str, shapely.Geometry] = {}
    with get_upath_local(country_data_path, extra_paths=aux_files) as local_fname:
        with fiona.open(local_fname) as src:
            for feat in src:
                country_name = feat["properties"]["ISO_A2"]
                if country_name not in countries:
                    continue
                cur_shp = shapely.geometry.shape(feat["geometry"])
                if country_name in country_wgs84_shps:
                    country_wgs84_shps[country_name] = country_wgs84_shps[
                        country_name
                    ].union(cur_shp)
                else:
                    country_wgs84_shps[country_name] = cur_shp

    return country_wgs84_shps


def process_shapes_into_events(
    tif_fname: str,
    shapes: list[shapely.Geometry],
    masked_date_data: npt.NDArray,
    projection: Projection,
    bounds: PixelBounds,
    country_wgs84_shps: dict[str, shapely.Geometry] | None,
    min_area: float,
) -> list[Feature]:
    """Process the forest loss shapes into vector features.

    Args:
        tif_fname: the GLAD tile filename.
        shapes: the shapes extracted from the forest loss mask.
        masked_date_data: the GLAD date raster, masked with the confidence and date
            constraints.
        projection: the projection of the pixel coordinates.
        bounds: the bounds of the pixel coordinates.
        country_wgs84_shps: optional dict mapping from country name to the country
            polygon in WGS84 coordinates. If set, only forest loss events in these
            countries will be returned, and the event properties will include a country
            field.
        min_area: minimum area constraint for each shape.
    """
    events: list[Feature] = []
    background_skip_count = 0
    area_skip_count = 0
    country_skip_count = 0

    for shp, value in tqdm.tqdm(shapes, desc="process shapes"):
        # Skip shapes corresponding to the background.
        if value != 1:
            background_skip_count += 1
            continue

        # Apply minimum area constraint, it must be in GLAD pixels which should be
        # 10 m/pixel.
        shp = shapely.geometry.shape(shp)
        if shp.area < min_area:
            area_skip_count += 1
            continue

        # Get center point (clipped to shape) and note the corresponding date.
        center_shp, _ = shapely.ops.nearest_points(shp, shp.centroid)
        center_pixel = (int(center_shp.x), int(center_shp.y))
        cur_days = int(masked_date_data[center_pixel[1], center_pixel[0]])

        if cur_days == 0:
            # Sometimes this can happen if the clipping was off a bit, and the
            # center_pixel is outside the connected component of alert pixels.
            continue

        cur_date = BASE_DATETIME + timedelta(days=cur_days)

        # Verify that the center is in the country polygon.
        center_src_geom = STGeometry(
            projection,
            shapely.Point(center_pixel[0] + bounds[0], center_pixel[1] + bounds[1]),
            (cur_date, cur_date),
        )
        center_wgs84_geom = center_src_geom.to_projection(WGS84_PROJECTION)
        matched_country: str | None = None
        if country_wgs84_shps is not None:
            for country_name, country_wgs84_shp in country_wgs84_shps.items():
                if not country_wgs84_shp.contains(center_wgs84_geom.shp):
                    continue
                matched_country = country_name
                break

            if matched_country is None:
                country_skip_count += 1
                continue

        # Translate shape to add the window topleft to the relative pixel coordinates,
        # yielding absolute coordinates. Here we also buffer the polygon to make sure
        # it is valid. We use quad_segs=4 to mitigate the growth in number of vertices
        # (the default is 8).
        translated_shp = shapely.affinity.translate(shp, xoff=bounds[0], yoff=bounds[1])
        translated_shp = shapely.buffer(translated_shp, distance=1, quad_segs=4)

        polygon_src_geom = STGeometry(
            projection,
            translated_shp,
            (cur_date, cur_date),
        )
        polygon_wgs84_geom = polygon_src_geom.to_projection(WGS84_PROJECTION)
        feat = Feature(
            polygon_wgs84_geom,
            properties=dict(
                center_pixel=center_pixel,
                tif_fname=tif_fname,
                oe_start_time=cur_date.isoformat(),
                oe_end_time=cur_date.isoformat(),
            ),
        )
        if matched_country is not None:
            feat.properties["country"] = matched_country
        events.append(feat)

    logger.debug(f"Skipped {background_skip_count} shapes as background")
    logger.debug(f"Skipped {area_skip_count} shapes due to area")
    logger.debug(f"Skipped {country_skip_count} shapes not in country polygon")
    return events


def extract_events_for_tile(
    args: ExtractAlertsArgs,
    tif_fname: str,
    country_wgs84_shps: dict[str, shapely.Geometry] | None,
) -> list[Feature]:
    """Extract vector features of forest loss events for the given GLAD alert tile.

    Args:
        args: the ExtractAlertsArgs.
        tif_fname: the GLAD alert tile filename to process.
        country_wgs84_shps: optional dict mapping from country names to the WGS84
            country polygons to limit events to.

    Returns:
        list of vector features.
    """
    # Read the confidence and date rasters from GCS, where they are published.
    # We also get the projection and bounds, which we can use for corodinate
    # transforms.
    conf_path = UPath(args.conf_prefix) / tif_fname
    logger.info(f"Read confidences from {conf_path}")
    with open_rasterio_upath_reader(conf_path) as src:
        conf_data = src.read(1)
        projection, bounds = get_raster_projection_and_bounds(src)

    date_path = UPath(args.date_prefix) / tif_fname
    logger.info(f"Read dates from {conf_path}")
    with open_rasterio_upath_reader(date_path) as src:
        date_data = src.read(1)

    # Now we compute the mask based on the confidence and date conditions.
    logger.info("Compute overall mask")
    now_days = (args.prediction_utc_time - BASE_DATETIME).days
    min_days = now_days - args.days
    date_mask = date_data >= min_days
    conf_mask = conf_data >= args.min_confidence
    forest_loss_mask = (date_mask & conf_mask).astype(np.uint8)

    if np.count_nonzero(forest_loss_mask) == 0:
        logger.warning(
            f"No forest loss events found for {tif_fname}, skipping further processing for this tile"
        )
        return []

    # Extract shapely geometries from the mask.
    logger.info(f"Create shapes from mask for {tif_fname}")
    shapes = list(rasterio.features.shapes(forest_loss_mask))

    # Finally we can process those shapes into forest loss events.
    # It requires a masked version of date_data, which we compute by multiplying
    # date_data by the constraints masked.
    masked_date_data = date_data * forest_loss_mask
    events = process_shapes_into_events(
        tif_fname=tif_fname,
        shapes=shapes,
        masked_date_data=masked_date_data,
        projection=projection,
        bounds=bounds,
        country_wgs84_shps=country_wgs84_shps,
        min_area=args.min_area,
    )

    # Limit to maximum number of events if desired.
    if (
        args.max_number_of_events is not None
        and len(events) > args.max_number_of_events
    ):
        logger.info(
            f"For tile {tif_fname}, limiting from {len(events)} to {args.max_number_of_events} events"
        )
        events = random.sample(events, args.max_number_of_events)

    logger.info(f"Got {len(events)} events for tile {tif_fname}")
    return events


def extract_alerts(
    extract_alerts_args: ExtractAlertsArgs,
) -> None:
    """Create a prediction request geometry based on GLAD alerts.

    Args:
        extract_alerts_args: the extract_alerts_args
    """
    logger.info(f"Extract_alerts for {str(extract_alerts_args)}")

    # Get country geometries to limit the area where we look for alerts.
    country_wgs84_shps: dict[str, shapely.Geometry] | None = None
    if extract_alerts_args.countries is not None:
        country_wgs84_shps = load_country_polygons(
            UPath(extract_alerts_args.country_data_path), extract_alerts_args.countries
        )

    # Process the GLAD alert tiles in parallel.
    # From each tile we extract some number of forest loss events.
    extract_events_for_tile_jobs = [
        dict(
            args=extract_alerts_args,
            tif_fname=tif_fname,
            country_wgs84_shps=country_wgs84_shps,
        )
        for tif_fname in extract_alerts_args.gcs_tiff_filenames
    ]
    p = multiprocessing.Pool(extract_alerts_args.workers)
    outputs = star_imap_unordered(
        p, extract_events_for_tile, extract_events_for_tile_jobs
    )

    all_events: list[Feature] = []
    for cur_events in tqdm.tqdm(outputs, total=len(extract_events_for_tile_jobs)):
        all_events.extend(cur_events)
    p.close()

    logger.info(f"Total events: {len(all_events)}")

    out_fname = UPath(extract_alerts_args.out_fname)
    out_fname.parent.mkdir(parents=True, exist_ok=True)
    GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84).encode_to_file(
        out_fname, all_events
    )
