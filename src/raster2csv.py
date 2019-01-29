from rasterio.windows import Window
import rasterio
import numpy as np
import math
import argparse


def raster2csv(input, output, separator, max_block_size):

    with rasterio.open(input) as src:

        affine = src.transform
        col_size = affine[0]
        row_size = affine[4]

        step_width, step_height = get_steps(src, max_block_size)

        first = True

        for col in range(0, src.width, step_width):
            for row in range(0, src.height, step_height):

                width = get_window_width(col, step_width, src.width)
                height = get_window_height(row, step_height, src.height)

                window = Window(col, row, width, height)

                left, top, right, bottom = src.window_bounds(window)
                w = src.read(1, window=window)

                if first:
                    table = get_xyz(w, col_size, row_size, left, bottom)
                    first = False
                else:
                    table = np.vstack((table, get_xyz(w, col_size, row_size, left, bottom)))

        np.savetxt(output, table, fmt='%.6f{0} %.6f{0} %{1}'.format(separator, get_date_type(src)))


def get_xyz(array, x_size, y_size, left, bottom):

    (y_index, x_index) = np.nonzero(array)

    x_coords = x_index * x_size + left + (x_size / 2)
    y_coords = y_index * y_size + bottom + (y_size / 2)

    lon = x_coords.reshape(len(x_coords), 1)
    lat = y_coords.reshape(len(y_coords), 1)

    mask = array > 0
    v = np.extract(mask, array)
    value = v.reshape(len(v), 1)

    return np.concatenate((lon, lat, value), axis=1)


def get_steps(image, max_size=4096):

    shape = image.block_shapes[0]
    step_width = math.floor(max_size/shape[0]) * shape[0]
    step_height = math.floor(max_size/shape[1]) * shape[1]

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
            "input",
            metavar='INPUT',
            type=str,
            help="Input Raster",
        )

        parser.add_argument(
            "output",
            metavar='OUTPUT',
            type=str,
            help="Output CSV",
        )

        parser.add_argument(
            "--separator",
            "-s",
            choices=[',', ';', 't'],
            type=str,
            help="Separator",
        )

        parser.add_argument(
            "--max_block_size",
            default=4096,
            type=int,
            help="max block size (multiple of 256)"
        )

        args = parser.parse_args()

        if args.separator=="t":
            separator = "\t"
        else:
            separator = args.seperator

        raster2csv(args.input, args.output, separator, args.max_block_size)


if __name__ == "__main__":
    main()


