from setuptools import setup  # , Extension


setup(
    name="raster2points",
    version="0.1.0",
    description="Tool to convert rasters to CSV files",
    package_dir={"": "src"},
    author="thomas.maschler",
    license="MIT",
    install_requires=["rasterio", "pandas", "numba"],
    scripts=["src/raster2csv.py"],
)
