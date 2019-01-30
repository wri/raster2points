from setuptools import setup  # , Extension


setup(
    name="raster2csv",
    version="0.0.1",
    description="Tool to convert rasters to CSV files",
    package_dir={"": "src"},
    py_modules=["raster2csv"],
    author="thomas.maschler",
    license="MIT",
    install_requires=["rasterio"],
    scripts=["src/raster2csv.py"],
)
