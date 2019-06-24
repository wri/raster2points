import argparse
from datetime import datetime
import logging
import sys

from rasterio.errors import RasterioIOError

from raster2points import raster2csv


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
    logger = logging.getLogger('raster2points')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt='%(asctime)s %(levelname)-4s %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description="Convert raster to CSV")

    parser.add_argument(
        "input", nargs="+", metavar="INPUT", type=str, help="Input Raster"
    )

    parser.add_argument("output", metavar="OUTPUT", type=str, help="Output CSV")

    parser.add_argument(
        "--col_names",
        default=None,
        nargs="+",
        type=str,
        help="Column names for raster values (default: val1, val2, ...)",
    )

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

    parser.add_argument(
        "--workers",
        "-w",
        default=1,
        type=int,
        help="Number of workers to run in parallel",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action='store_true',
        help="Print additional logging",
    )

    args = parser.parse_args()

    if args.separator == "t":
        separator = "\t"
    else:
        separator = args.separator

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    files = args.input + [args.output]
    start = datetime.now()
    try:
        raster2csv(
            *files,
            col_names=args.col_names,
            separator=separator,
            max_block_size=args.max_block_size,
            calc_area=args.calc_area,
            workers=args.workers
        )
    except (AssertionError, RasterioIOError) as e:
        logger.error(e, exc_info=logger.getEffectiveLevel() == logging.DEBUG)
        sys.exit(1)
    logger.info("time elapsed: {}".format(datetime.now() - start))


if __name__ == "__main__":

    main()
