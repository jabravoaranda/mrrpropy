import xarray as xr
from matplotlib import pyplot as plt

data_path = "sliding_info_full_hour.nc"
ds = xr.open_dataset(data_path)

plt.hist(ds.where(ds["proc_label"] == "activation")["Dm_layer_mean"].values.flatten(), bins=20)
plt.xlabel("Dm_layer_mean")
plt.ylabel("Frequency")
plt.title("Distribution of Dm_layer_mean for Activation Events")
plt.show()