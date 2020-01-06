# raster2points

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/af460e23844a48b9ab0502a362e7ec10)](https://www.codacy.com/gh/wri/raster2points?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=wri/raster2points&amp;utm_campaign=Badge_Grade)

Convert one or multiple raster images to points.
Tool will read first input raster and extract lat/lon coordinates and values
for all pixels which have data. Optional it calculates geodesic area for each point based on pixel size.
Successive input rasters will use data mask from first input raster.

Returns a Pandas dataframe, CLI will export results as CSV file.

Input files can be local file paths or S3 paths, or a mix. For reading from
S3, you'll need [AWS credentials configured](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html),
such as with a profile in `~/.aws` and an `AWS_PROFILE` variable in your environment.

Multi-worker only works with S3 inputs, not local files.

## Installation and Dependencies

This module uses rasterio and requires `GDAL>=1.11`.
Use pip to install.

```bash
pip install raster2points
```

## CLI Usage
```bash
raster2csv.py   [-h]
                [--col_names COL_NAMES [COL_NAMES ...]]
                [--separator {,,;,t}]
                [--max_block_size MAX_BLOCK_SIZE]
                [--calc_area [CALC_AREA]]
                [--workers WORKERS]
                INPUT [INPUT ...]
                OUTPUT

```

## Python Usage
You can also use the module directly in python. It will return a
Pandas dataframe with your data.

Get Pandas data frame
```python
from raster2points import raster2df

raster1 = "path/to/file1.tif"
raster2 = "path/to/file2.tif"

df = raster2df(raster1, raster2, col_names=["name1", "name2"], calc_area=True)

print(df.columns)
print(df.dtypes)

df.head()
```

Export to TSV
```python
from raster2points import raster2csv

raster1 = "path/to/file1.tif"
raster2 = "path/to/file2.tif"
output = "path/to/newfile.tsv"

raster2csv(raster1, raster2, output, col_names=["name1", "name2"], separator="\t", calc_area=True)
```
