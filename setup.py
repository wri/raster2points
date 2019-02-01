from setuptools import setup  # , Extension


setup(
    name="raster2points",
    version="0.1.0",
    description="Tool to convert rasters to points",
    package_dir={"": "src"},
    author="Thomas Maschler",
    author_email="thomas.maschler@wri.org",
    license="MIT",
    install_requires=["rasterio", "pandas", "numba"],
    scripts=["src/raster2csv.py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
