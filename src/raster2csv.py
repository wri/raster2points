import argparse
from raster2points import raster2df
from datetime import datetime


def raster2csv(src_rasters, csv_file, separator, max_block_size=4096, calc_area=False):
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

    table = raster2df(src_rasters, max_block_size, calc_area)

    table.to_csv(csv_file, sep=separator, header=True, index=False)


def str2bool(v):
    """
    Convert various strings to boolean
    :param v: String
    :return: Boolean
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
        "--calc_area",
        type=str2bool,
        nargs="?",
        default=False,
        const=True,
        help="Calculate Pixel geodesic area",
    )

    args = parser.parse_args()

    if args.separator == "t":
        separator = "\t"
    else:
        separator = args.separator

    raster2csv(args.input, args.output, separator, args.max_block_size, args.calc_area)


if __name__ == "__main__":
    now = datetime.now()
    main()
    print(datetime.now() - now)
