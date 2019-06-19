import math
import itertools
import logging

from numba import njit
import numpy as np
import pandas as pd
from parallelpipe import Stage
import rasterio
from rasterio.windows import Window

logger = logging.getLogger('raster2points')


def raster2csv(
    *files,
    col_names=None,
    separator=",",
    max_block_size=4096,
    calc_area=False,
    workers=1
):
    """
    Convert rasters to CSV.
    Input rasters must match cell size and extent.
    Tool writes final result text file
    :param src_rasters: list of input rasters
    :param csv_file: output file
    :param separator: separator used in CSV file
    :param max_block_size: max block size to process
    :return: None
    """
    assert len(files) >= 2, "No output file provided"

    csv_file = files[-1]
    src_rasters = files[:-1]

    logger.info(
        "Extract data using {} worker{}".format(workers, "" if workers == 1 else "s")
    )
    table = raster2df(
        *src_rasters,
        col_names=col_names,
        max_block_size=max_block_size,
        calc_area=calc_area,
        workers=workers,
    )

    logger.info("Write to file: " + csv_file)
    table.to_csv(csv_file, sep=separator, header=True, index=False)

    logger.info("Done.")


def raster2df(
    *src_rasters, col_names=None, max_block_size=4096, calc_area=False, workers=1
):
    """
    Converts raster into Panda DataFrame.
    Input rasters must match cell size and extent.
    The first raster determines number of output rows.
    Only cells which are are above given Threshold/ not NoData are processed
    The tool calculates lat lon for every grid cell and extract the cell value.
    If more than one input raster is provided tool adds additional columns to CSV with coresponing values.
    :param src_rasters: Input rasters (one or many)
    :param col_names: Column names for input raster values (optional, default: val1, val2, ...)
    :param max_block_size: maximum block size to process in at once
    :param calc_area: Calculate geodesic area
    :param workers: number of parallel workers
    :return: Pandas data frame
    """

    if col_names:
        assert len(src_rasters) == len(
            col_names
        ), "Number of named columns does not match number of input rasters. Abort."

    sources = _assert_sources(src_rasters)

    src = sources[0]
    affine = src.transform
    step_height, step_width = _get_steps(src, max_block_size)

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

    pipe = blocks | Stage(_process_blocks, sources, **kwargs).setup(workers=workers)

    data_frame = pd.DataFrame()
    for df in pipe.results():
        if data_frame.empty:
            data_frame = df[0]  # unpack data frame from tuple
        else:
            data_frame = pd.concat([data_frame, df[0]])

    logger.debug("Renaming columns")
    if col_names:
        i = 0
        for col_name in col_names:
            data_frame = data_frame.rename(
                index=str, columns={"val{}".format(i): col_name}
            )
            i += 1

    for src in sources:
        src.close()

    return data_frame


def _assert_sources(src_rasters):
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
            assert width == src.width, "Input rasters must have same dimensions. Abort."
            assert (
                height == src.height
            ), "Input rasters must have same dimensions Abort."
            s_left, s_bottom, s_right, s_top = src.bounds
            assert round(left, 4) == round(
                s_left, 4
            ), "Input rasters must have same bounds. Abort."
            assert round(bottom, 4) == round(
                s_bottom, 4
            ), "Input rasters must have same bounds. Abort."
            assert round(right, 4) == round(
                s_right, 4
            ), "Input rasters must have same bounds. Abort."
            assert round(top, 4) == round(
                s_top, 4
            ), "Input rasters must have same bounds. Abort."
        sources.append(src)

    return sources


def _process_blocks(
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

        logger.debug("Processing block ({}, {})".format(col, row))

        w_width = _get_window_size(col, step_width, width)
        w_height = _get_window_size(row, step_height, height)
        window = Window(col, row, w_width, w_height)

        src = sources[0]

        left, top, right, bottom = src.window_bounds(window)
        w = src.read(1, window=window)
        lat_lon = _get_lat_lon(w, col_size, row_size, left, bottom, calc_area)
        del w

        if lat_lon.shape[0] > 0:

            values = _get_values(sources, window)

            yield (
                pd.concat([lat_lon, values], axis=1),
            )  # need to pack data frame into tuple


def _get_lat_lon(array, x_size, y_size, left, bottom, calc_area):
    """
    Create x/y indices for all nonzero pixels
    Computes lat lon coordinates based on lower left corner and pixel size

    :param array: Numpy Array for given image
    :param x_size: pixel width
    :param y_size: pixel height
    :param left: min lon
    :param bottom: min lat
    :return: Pandas data frame with lat/lon coordinates
    """
    (y_index, x_index) = _get_index(array)

    x_coords = _get_coord(x_index, x_size, left)
    y_coords = _get_coord(y_index, y_size, bottom)

    lon = pd.Series(x_coords)
    lat = pd.Series(y_coords)

    df = pd.DataFrame(
        {
            "lon": lon.astype("float32", copy=False),
            "lat": lat.astype("float32", copy=False),
        }
    )

    if calc_area:
        area = pd.Series(_get_area(y_coords, y_size, x_size)).astype(
            "float32", copy=False
        )
        df["area"] = area

    return df


def _get_values(sources, window, threshold=0):
    """
    Extract values for all non null cells for all input images
    :param src_rasters:
    :param window:
    :param threshold:
    :return: Pandas Dataframe with values
    """
    df_col = 0
    for src in sources:
        dtype = src.dtypes[0]
        w = src.read(1, window=window)

        if df_col == 0:
            mask = _get_mask(w, threshold)

        s = pd.Series(_apply_mask(mask, w))

        if df_col == 0:
            df = pd.DataFrame({"val{}".format(df_col): s.astype(dtype, copy=False)})

        else:
            df["val{}".format(df_col)] = s.astype(dtype, copy=False)

        df_col += 1

    return df


@njit()  # using numba.jit to precompile calculations
def _get_mask(w, threshold):
    return w > threshold


@njit()  # using numba.jit to precompile calculations
def _apply_mask(mask, w):
    return np.extract(mask, w)


@njit()  # using numba.jit to precompile calculations
def _get_index(array):
    return np.nonzero(array)


@njit()  # using numba.jit to precompile calculations
def _get_coord(index, size, offset):
    return index * size + offset + (size / 2)


@njit()  # using numba.jit to precompile calculations
def _get_area(lat, d_lat, d_lon):
    """
    Calculate geodesic area for grid cells using WGS 1984 as spatial reference.
    Cell/Pixel size various with latitude.
    :param lat: array with lat coord
    :param d_lat: pixel hight
    :param d_lon: pixel width
    :return: Numpy Array with pixel area
    """

    a = 6378137.0  # Semi major axis of WGS 1984 ellipsoid
    b = 6356752.314245179  # Semi minor axis of WGS 1984 ellipsoid

    pi = math.pi

    q = d_lon / 360
    e = math.sqrt(1 - (b / a) ** 2)

    area = (
        np.abs(
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


def _get_steps(image, max_size=4096):
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

    # stripped image, each block represents one row
    if shape[1] == image.width:
        max_size = max_size ** 2

        step_width = shape[1]
        if shape[0] * shape[1] > max_size:
            step_height = shape[0]
        else:
            step_height = math.floor(max_size / shape[1] / shape[0]) * shape[0]

    # tiled image, blocks width and height have equal size
    else:
        if shape[0] * shape[1] > max_size ** 2:
            step_width = shape[1]
            step_height = shape[0]

        else:
            step_width = math.floor(max_size / shape[1]) * shape[1]
            step_height = math.floor(max_size / shape[0]) * shape[0]

    return step_height, step_width


def _get_window_size(offset, step_size, image_size):
    """
    Calculate window width or height.
    Usually same as block size, except when at the end of image and only a
    fracture of block size remains
    :param offset: start columns/ row
    :param step_size: block width/ height
    :param image_size: image width/ height
    :return: window width/ height
    """
    if offset + step_size > image_size:
        return image_size - offset
    else:
        return step_size
