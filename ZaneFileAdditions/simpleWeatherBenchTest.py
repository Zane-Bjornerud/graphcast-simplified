import xarray as xr
import gcsfs

# Create filesystem
fs = gcsfs.GCSFileSystem(token="anon")

print("pre-path")
# The path from your screenshot
path = "gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours-64x32_equiangular_conservative.zarr"

print("pre-open")
# Open (doesn't download anything yet!)
ds = xr.open_zarr(path)

# Explore
print("Available variables:", list(ds.data_vars))
print("Dimensions:", ds.dims)
print("Time range:", ds.time.values[0], "to", ds.time.values[-1])
print("Total timesteps:", len(ds.time))
