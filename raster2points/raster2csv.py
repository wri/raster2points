import argparse
from raster2points import raster2csv
from rasterio.errors import RasterioIOError
import logging
import sys
from datetime import datetime


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

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

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

    parser.add_argument(
        "--workers",
        "-w",
        default=1,
        type=int,
        help="Number of workers to run in parallel",
    )

    args = parser.parse_args()

    if args.separator == "t":
        separator = "\t"
    else:
        separator = args.separator

    files = args.input + [args.output]

    start = datetime.now()
    try:
        raster2csv(
            *files,
            separator=separator,
            max_block_size=args.max_block_size,
            calc_area=args.calc_area,
            workers=args.workers
        )
    except (AssertionError, RasterioIOError) as e:
        logging.error(e, exc_info=logger.getEffectiveLevel() == logging.DEBUG)
        sys.exit(1)
    logging.info("time elapsed: {}".format(datetime.now() - start))


if __name__ == "__main__":

    main()
