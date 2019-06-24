from setuptools import setup

with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name="raster2points",
    version="0.1.5",
    description="Tool to convert rasters to points",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wri/raster2points",
    packages=["raster2points"],
    author="Thomas Maschler",
    author_email="thomas.maschler@wri.org",
    license="MIT",
    install_requires=[
        "numba~=0.44.1",
        "pandas ~=0.24.2",
        "parallelpipe~=0.2.6",
        "pre-commit~=1.17.0",
        "pytest~=4.6.3",
        "rasterio[s3]~=1.0.24",
    ],
    scripts=["raster2points/raster2csv.py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
