from rasterio.windows import Window
import rasterio
import numpy as np
import pandas as pd
import math
import argparse
import itertools
import multiprocessing

# from parallelpipe import Stage
from datetime import datetime


def raster2csv(src_rasters, csv_file, separator, max_block_size, workers):
    """
    Convert rasters to CSV. Run in parallel by reading blocks.
    The first raster determines number of output rows.
    Only cells which are are above given Threshold/ not NoData are processed
    The tool calculates lat lon for every grid cell and extract the cell value.
    If more than one input raster is provided tool adds additional columns to CSV with coresponing values.
    Input rasters must match cell size and extent.
    Tool writes final result text file
    :param src_rasters: list of input rasters
    :param csv_file: output file
    :param separator: separator used in CSV file
    :param max_block_size: max block size to process
    :param workers: number of parallel processes
    :return: None
    """

    with rasterio.open(src_rasters[0]) as src:

        affine = src.transform
        step_width, step_height = get_steps(src, max_block_size)

        kwargs = {
            "col_size": affine[0],
            "row_size": affine[4],
            "step_width": step_width,
            "step_height": step_height,
            "width": src.width,
            "height": src.height,
        }

        cols = range(0, src.width, step_width)
        rows = range(0, src.height, step_height)

    blocks = itertools.product(cols, rows)

    # pipe = blocks | Stage(process_blocks, src_rasters, **kwargs).setup(workers=workers)
    # Tried using parallel pipeline to speed up processing
    # This throws and error b/c truth numpy arrays and pandas data frames are ambiguous.
    # Submitted ticket here https://github.com/gtsystem/parallelpipe/issues/1
    # Not sure if this is something parallelpipe can fix or if this is multiprocssing issue
    dfs = process_blocks(blocks, src_rasters, **kwargs)

    first = True
    for df in dfs:

        if first:
            table = df
            first = False
        else:

            table = pd.concat([table, df])

    table.to_csv(csv_file, sep=separator, header=True, index=False)


def process_blocks(
    blocks, src_rasters, col_size, row_size, step_width, step_height, width, height
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

        with rasterio.open(src_rasters[0]) as src:
            left, top, right, bottom = src.window_bounds(window)
            w = src.read(1, window=window)
            lat_lon = get_lat_lon(w, col_size, row_size, left, bottom)
        del w

        if lat_lon.shape[0] == 0:
            break

        values = get_values(src_rasters, window)

        yield pd.concat([lat_lon, values], axis=1)


def get_lat_lon(array, x_size, y_size, left, bottom):
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
    (y_index, x_index) = np.nonzero(array)

    x_coords = x_index * x_size + left + (x_size / 2)
    y_coords = y_index * y_size + bottom + (y_size / 2)

    lon = pd.Series(x_coords)  # x_coords.reshape(len(x_coords), 1)
    lat = pd.Series(y_coords)  # y_coords.reshape(len(y_coords), 1)

    df = pd.DataFrame(
        {
            "lon": lon.astype("float32", copy=False),
            "lat": lat.astype("float32", copy=False),
        }
    )

    return df  # np.concatenate((lon, lat), axis=1)


def get_values(src_rasters, window, threshold=0):
    """
    Extract values for all non null cells for all input images
    :param src_rasters:
    :param window:
    :param threshold:
    :return:
    """
    df_col = 0
    for raster in src_rasters:
        with rasterio.open(raster) as src:
            dtype = src.dtypes[0]
            w = src.read(1, window=window)

        if df_col == 0:
            mask = w > threshold

        s = pd.Series(np.extract(mask, w))

        if df_col == 0:
            df = pd.DataFrame({"val{}".format(df_col): s.astype(dtype, copy=False)})
            # values = v.reshape(len(v), 1)
            # first = False
        else:
            df["val{}".format(df_col)] = s.astype(dtype, copy=False)
            # values = np.concatenate((values, v.reshape(len(v), 1)), axis=1)
        df_col += 1

    return df


def get_steps(image, max_size=4096):
    """
    Compute optimal block size.
    Should be always a multiple of image block size
    :param image: image to process
    :param max_size: maximal block size
    :return: block width and height
    """
    shape = image.block_shapes[0]
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


def get_date_type(image):
    """
    Get data type of current image to correctly format CSV
    :param image:
    :return:
    """
    if "int" in image.dtypes[0]:
        return "d"
    else:
        return "6f"


def main():

    parser = argparse.ArgumentParser(description="Convert raster to CSV")

    parser.add_argument(
        "input", nargs="+", metavar="INPUT", type=str, help="Input Raster"
    )

    parser.add_argument("output", metavar="OUTPUT", type=str, help="Output CSV")

    parser.add_argument(
        "--separator",
        "-s",
        default=",",
        choices=[",", ";", "t"],
        type=str,
        help="Separator",
    )

    parser.add_argument(
        "--max_block_size",
        default=4096,
        type=int,
        help="max block size (multiple of 256)",
    )

    parser.add_argument(
        "--workers",
        "-w",
        default=multiprocessing.cpu_count(),
        type=int,
        help="Number of parallel processes",
    )

    args = parser.parse_args()

    if args.separator == "t":
        separator = "\t"
    else:
        separator = args.seperator

    raster2csv(args.input, args.output, separator, args.max_block_size, args.workers)


if __name__ == "__main__":
    now = datetime.now()
    main()
    print(datetime.now() - now)
