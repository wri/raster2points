# raster2points

Convert one or multiple raster images to points.
Tool will read first input raster and extract lat/lon coordinates and values
for all pixels which have data. Optional it calculates geodesic area for each point based on pixel size.
Successive input rasters will use data mask from first input raster.

Returns a Pandas dataframe, CLI will export results as CSV file.


## Installation and Dependencies

This module uses rasterio and requires `GDAL>=1.11`.
It is not yet available on pypy but you can install it directly from within the repo.

```bash
pip install -e .
```

## CLI Usage:
```bash
raster2csv.py [-h] [--separator {,,;,t}] [--max_block_size MAX_BLOCK_SIZE] [--calc_area] INPUT [INPUT ...] OUTPUT

```

## Python Usage
You can also use the module directly in python. It will return a
Pandas dataframe with your data.

```python
from raster2points import raster2df

raster1 = "path/to/file1"
raster2 = "path/to/file2"

df = raster2df(raster1, raster2, calc_area=True)

print(df.columns)
print(df.dtypes)

df.head()
```
