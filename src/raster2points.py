from rasterio.windows import Window
import rasterio
import numpy as np
import pandas as pd
import math
import itertools
from numba import jit


def raster2df(src_rasters, max_block_size=4096, calc_area=False):
    """
    Converts raster into Panda DataFrame.
    Input rasters must match cell size and extent.
    The first raster determines number of output rows.
    Only cells which are are above given Threshold/ not NoData are processed
    The tool calculates lat lon for every grid cell and extract the cell value.
    If more than one input raster is provided tool adds additional columns to CSV with coresponing values.
    :param src_rasters:
    :param max_block_size:
    :return:
    """

    sources = list()
    first = True
    for raster in src_rasters:

        src = rasterio.open(raster)
        if first:
            width = src.width
            height = src.height
            left, bottom, right, top = src.bounds
            first = False
        else:
            assert width == src.width, "Input images have different width"
            assert height == src.height, "Input images have different height"
            s_left, s_bottom, s_right, s_top = src.bounds
            assert round(left, 4) == round(
                s_left, 4
            ), "Input images have different bounds"
            assert round(bottom, 4) == round(
                s_bottom, 4
            ), "Input images have different bounds"
            assert round(right, 4) == round(
                s_right, 4
            ), "Input images have different bounds"
            assert round(top, 4) == round(
                s_top, 4
            ), "Input images have different bounds"
        sources.append(src)

    src = sources[0]
    affine = src.transform
    step_width, step_height = get_steps(src, max_block_size)

    kwargs = {
        "col_size": affine[0],
        "row_size": affine[4],
        "step_width": step_width,
        "step_height": step_height,
        "width": src.width,
        "height": src.height,
        "calc_area": calc_area,
    }

    cols = range(0, src.width, step_width)
    rows = range(0, src.height, step_height)

    blocks = itertools.product(cols, rows)

    # pipe = blocks | Stage(process_blocks, src_rasters, **kwargs).setup(workers=workers)
    # Tried using parallel pipeline to speed up processing
    # This throws and error b/c truth numpy arrays and pandas data frames are ambiguous.
    # Submitted ticket here https://github.com/gtsystem/parallelpipe/issues/1
    # Not sure if this is something parallelpipe can fix or if this is multiprocssing issue
    # TODO: This works when packing the array into a tuple (array,)
    #  still need to reimplemente later

    dfs = process_blocks(blocks, sources, **kwargs)

    first = True
    for df in dfs:

        if first:
            dataframe = df
            first = False
        else:

            dataframe = pd.concat([dataframe, df])

    for src in sources:
        src.close()

    return dataframe


def process_blocks(
    blocks,
    sources,
    col_size,
    row_size,
    step_width,
    step_height,
    width,
    height,
    calc_area,
):
    """
    Loops over all blocks and reads first input raster to get coordinates.
    Append values from all input rasters
    :param blocks: list of blocks to process
    :param src_rasters: list of input rasters
    :param col_size: pixel width
    :param row_size: pixel height
    :param step_width: block width
    :param step_height: block height
    :param width: image width
    :param height: image height
    :return: Table of Lat/Lon coord and corresponding raster values
    """

    for block in blocks:

        col = block[0]
        row = block[1]

        w_width = get_window_width(col, step_width, width)
        w_height = get_window_height(row, step_height, height)
        window = Window(col, row, w_width, w_height)

        src = sources[0]

        left, top, right, bottom = src.window_bounds(window)
        w = src.read(1, window=window)
        lat_lon = get_lat_lon(w, col_size, row_size, left, bottom, calc_area)
        del w

        if lat_lon.shape[0] == 0:
            break

        values = get_values(sources, window)

        yield pd.concat([lat_lon, values], axis=1)


def get_lat_lon(array, x_size, y_size, left, bottom, calc_area):
    """
    Create x/y indices for all nonzero pixels
    Computes lat lon coordinates based on lower left corner and pixel size

    :param array: Numpy Array for given image
    :param x_size: pixel width
    :param y_size: pixel height
    :param left: min lon
    :param bottom: min lat
    :return: Table of lat/lon cooridinates
    """
    (y_index, x_index) = get_index(array)

    x_coords = get_coord(x_index, x_size, left)
    y_coords = get_coord(y_index, y_size, bottom)

    lon = pd.Series(x_coords)
    lat = pd.Series(y_coords)

    df = pd.DataFrame(
        {
            "lon": lon.astype("float32", copy=False),
            "lat": lat.astype("float32", copy=False),
        }
    )

    if calc_area:
        area = pd.Series(get_area(y_coords, y_size, x_size)).astype(
            "float32", copy=False
        )
        df["area"] = area

    return df


def get_values(sources, window, threshold=0):
    """
    Extract values for all non null cells for all input images
    :param src_rasters:
    :param window:
    :param threshold:
    :return:
    """
    df_col = 0
    for src in sources:
        dtype = src.dtypes[0]
        w = src.read(1, window=window)

        if df_col == 0:
            mask = get_mask(w, threshold)

        s = pd.Series(apply_mask(mask, w))

        if df_col == 0:
            df = pd.DataFrame({"val{}".format(df_col): s.astype(dtype, copy=False)})

        else:
            df["val{}".format(df_col)] = s.astype(dtype, copy=False)

        df_col += 1

    return df


@jit()  # using numba.jit to precompile calculations
def get_mask(w, threshold):
    return w > threshold


@jit()  # using numba.jit to precompile calculations
def apply_mask(mask, w):
    return np.extract(mask, w)


@jit()  # using numba.jit to precompile calculations
def get_index(array):
    return np.nonzero(array)


@jit()  # using numba.jit to precompile calculations
def get_coord(index, size, offset):
    return index * size + offset + (size / 2)


@jit()  # using numba.jit to precompile calculations
def get_area(lat, d_lat, d_lon):
    """
    Calculate geodesic area for grid cells using WGS 1984 as spatial reference.
    Cell/Pixel size various with latitude.
    """
    a = 6378137.0  # Semi major axis of WGS 1984 ellipsoid
    b = 6356752.314245179  # Semi minor axis of WGS 1984 ellipsoid

    pi = math.pi

    q = d_lon / 360
    e = math.sqrt(1 - (b / a) ** 2)

    area = (
        abs(
            (
                pi
                * b ** 2
                * (
                    2 * np.arctanh(e * np.sin(np.radians(lat + d_lat))) / (2 * e)
                    + np.sin(np.radians(lat + d_lat))
                    / (
                        (1 + e * np.sin(np.radians(lat + d_lat)))
                        * (1 - e * np.sin(np.radians(lat + d_lat)))
                    )
                )
            )
            - (
                pi
                * b ** 2
                * (
                    2 * np.arctanh(e * np.sin(np.radians(lat))) / (2 * e)
                    + np.sin(np.radians(lat))
                    / (
                        (1 + e * np.sin(np.radians(lat)))
                        * (1 - e * np.sin(np.radians(lat)))
                    )
                )
            )
        )
        * q
    )

    return area


def get_steps(image, max_size=4096):
    """
    Compute optimal block size.
    Should be always a multiple of image block size
    Only if block size is bigger than maximal allowed step size,
    value will be forced to max_size
    :param image: image to process
    :param max_size: maximal step size
    :return: step width and height
    """
    shape = image.block_shapes[0]
    if shape[0] > max_size:
        shape[0] = max_size
    if shape[1] > max_size:
        shape[1] = max_size

    step_width = math.floor(max_size / shape[0]) * shape[0]
    step_height = math.floor(max_size / shape[1]) * shape[1]

    return step_width, step_height


def get_window_width(col, step_width, image_width):
    """
    Calculate window width.
    Usually same as block size, except when at the end of image and only a
    fracture of block size remains
    :param col: start columns
    :param step_width: block width
    :param image_width: image width
    :return: window width
    """
    if col + step_width > image_width:
        return image_width - col
    else:
        return step_width


def get_window_height(row, step_height, image_height):
    """
        Calculate window height.
        Usually same as block size, except when at the end of image and only a
        fracture of block size remains
        :param row: start row
        :param step_height: block height
        :param image_height: image height
        :return: window height
        """
    if row + step_height > image_height:
        return image_height - row
    else:
        return step_height
