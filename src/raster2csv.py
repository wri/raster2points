from rasterio.windows import Window
import rasterio
import numpy as np
import math
import argparse
import itertools
from parallelpipe import Stage
from datetime import datetime
import multiprocessing


def raster2csv(src_rasters, csv_file, separator, max_block_size, workers):

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

    pipe = blocks | Stage(compute_blocks, src_rasters, **kwargs).setup(workers=workers)

    first = True
    for output in pipe.results():
        output = np.asarray(output)

        if first:
            table = output
            first = False
        else:

            table = np.vstack((table, output))

    np.savetxt(
        csv_file,
        table,
        fmt="%.6f{0} %.6f{0} %{1}".format(separator, get_date_type(src)),
    )


def compute_blocks(
    blocks, src_rasters, col_size, row_size, step_width, step_height, width, height
):

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

        yield np.concatenate((lat_lon, values), axis=1).tolist()


def get_lat_lon(array, x_size, y_size, left, bottom):
    (y_index, x_index) = np.nonzero(array)

    x_coords = x_index * x_size + left + (x_size / 2)
    y_coords = y_index * y_size + bottom + (y_size / 2)

    lon = x_coords.reshape(len(x_coords), 1)
    lat = y_coords.reshape(len(y_coords), 1)

    return np.concatenate((lon, lat), axis=1)


def get_value(array, threshold=0):

    mask = array > threshold
    v = np.extract(mask, array)
    return v.reshape(len(v), 1)


def get_values(src_rasters, window, threshold=0):

    first = True
    for raster in src_rasters:
        with rasterio.open(raster) as src:
            w = src.read(1, window=window)

            if first:
                mask = w > threshold
            v = np.extract(mask, w)

            if first:
                values = v.reshape(len(v), 1)
                first = False
            else:
                np.concatenate((values, v.reshape(len(v), 1)), axis=1)

    return values


def get_steps(image, max_size=4096):

    shape = image.block_shapes[0]
    step_width = math.floor(max_size / shape[0]) * shape[0]
    step_height = math.floor(max_size / shape[1]) * shape[1]

    return step_width, step_height


def get_window_width(col, step_width, image_width):
    if col + step_width > image_width:
        return image_width - col
    else:
        return step_width


def get_window_height(row, step_height, image_height):
    if row + step_height > image_height:
        return image_height - row
    else:
        return step_height


def get_date_type(image):
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
