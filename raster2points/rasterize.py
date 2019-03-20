from rasterio import features
from shapely.geometry import shape
import rasterio


def get_bounds(geojson):
    """
    :param geojson: a GEOJSON like Python object
    :return: (minx, miny, maxx, maxy)
    """
    geom = shape(geojson)
    return geom.bounds


def get_shape(bounds, pixel_with, pixel_height):
    pass


def build_mask(geojson, pixel_width, pixel_height):
    bounds = get_bounds(geojson)
    shape = get_shape(bounds, pixel_width, pixel_height)
    mask = features.rasterize(
        geojson, out_shape=shape, fill=0, default_value=1, dtype=rasterio.uint8
    )
    return mask
