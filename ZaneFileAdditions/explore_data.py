import xarray as xr
import numpy as np
import matplotlib.pyplot as plt  # for visualization

xr.set_options(
    display_width=200,  # Wider display
    display_max_rows=100,  # Show more rows
    display_expand_attrs=True,  # Show all attributes
    display_expand_data=True,  # Show full data info
    display_expand_coords=True,  # Show all coordinates
    display_expand_data_vars=True,  # Show all data variables
)

## GEOPOTENTIAL DATA ##
filePathGP = "../weatherBench2VarData/geopotential_500_5.625deg/geopotential_500hPa_1979_5.625deg.nc"

print("Oepning GeoP file")
fGP = xr.open_dataset(filePathGP)
print("Geopotential Dataset @ 5.625 deg")
print(fGP)

data_varGP = "z"
dataGP = fGP[data_varGP]
print("Visualizing variable ", data_varGP)
print("Shape: ", dataGP.shape)
first_timestepGP = dataGP[0, :, :]
plt.figure(figsize=(12, 6))
plt.imshow(first_timestepGP, cmap="RdBu_r", aspect="auto")
plt.colorbar(label=f"{data_varGP}")
plt.title(f"{data_varGP} at {fGP.time.values[0]}")
plt.xlabel("Longitude Index")
plt.ylabel("Latitude Index")
plt.savefig(f"{data_varGP}_visualization.png")


## TEMPERATURE DATA ##
filePathTemp = (
    "../weatherBench2VarData/2m_temperature_5.625deg/2m_temperature_1979_5.625deg.nc"
)

print("Oepning Temp file")
fTemp = xr.open_dataset(filePathTemp)
print("Temp Dataset @ 5.625 deg")
print(fTemp)

data_varT = "t2m"
dataT = fTemp[data_varT]
print("Visualizaing variable ", data_varT)
print("Shape: ", dataT.shape)
first_timestepT = dataT[0, :, :]
plt.figure(figsize=(12, 6))
plt.imshow(first_timestepT, cmap="RdBu_r", aspect="auto")
plt.colorbar(label=f"{data_varT}")  # t2m
plt.title(f"{data_varT} at {fTemp.time.values[0]}")
plt.xlabel("Longitude Index")
plt.ylabel("Latitude Index")
plt.savefig(f"{data_varT}_visualization.png")
