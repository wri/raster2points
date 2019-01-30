# raster2csv

Convert one or multiple raster images to CSV files.
Tool will read first input raster and extract lat/lon coordinates and values 
for all pixels which have data. Successive input rasters will use data mask 
from first input raster.

Output file is a CSV file.

##Usage:
```bash
raster2csv.py [-h] [--separator {,,;,t}] [--max_block_size MAX_BLOCK_SIZE] INPUT [INPUT ...] OUTPUT

``` 

##Installation and Dependencies

This module use rasterio and requires `GDAL>=1.11` to be installed.
This module is not yet available on pypy 
but you can install it directly from within the repo

```bash
pip install -e .
```

## Usage with python
You can also use the module directly in python. It will return a 
Pandas dataframe with your data.

```python
import raster2csv

raster1 = "path/to/file1"
raster2 = "path/to/file2"

df = raster2csv.raster2df(raster1, raster2)

print(df.columns)
print(df.dtypes)

df.head()

```
