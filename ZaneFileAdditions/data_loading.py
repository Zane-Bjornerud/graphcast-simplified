import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

# CLEAR EVERYTHING
import sys
import gc

# Force garbage collection
gc.collect()

# Clear NumPy cache
if hasattr(np, "_clearCache"):
    np._clearCache()


def load_combined_data():
    # load geopotential and temperature data, combine into one array, and return that combined array

    gp_array = []
    t_array = []
    time_coords = []

    folderGP = Path("./weatherBench2VarData/geopotential_500_5.625deg")
    folderT = Path("./weatherBench2VarData/2m_temperature_5.625deg")

    for file in folderGP.rglob("*.nc"):
        dataGP = xr.open_dataset(file)
        gp_array.append(dataGP["z"].values)

        lat_coord = dataGP.lat.values
        lon_coord = dataGP.lon.values

        # Collect time coordinates from each file
        time_coords.append(dataGP.time.values)

        dataGP.close()

    # print("Done appending GP data. Array length is: ", len(gp_array))

    for file in folderT.rglob("*.nc"):
        dataT = xr.open_dataset(file)
        t_array.append(dataT["t2m"].values)
        dataT.close()

    # print("Done appending Temperature data. Array length is: ", len(t_array))

    gp_all = np.concatenate(gp_array, axis=0)
    # print("total all gp data: ", len(gp_all))

    t_all = np.concatenate(t_array, axis=0)
    # print("total all temp data: ", len(t_all))

    # Concatenate time coordinates
    time_all = np.concatenate(time_coords, axis=0)
    # print(f"Time coordinate length: {len(time_all)}")

    # Verify shapes match
    if gp_all.shape != t_all.shape:
        raise ValueError(f"Shape mismatch! GP: {gp_all.shape}, T: {t_all.shape}")

    # Stack into [time, lat, lon, 2]
    # print("\nStacking variables...")
    combined = np.stack([gp_all, t_all], axis=-1)

    # print(f"Combined shape: {combined.shape}")
    # print(f"Total elements: {combined.size:,}")

    coords = {"time": time_all, "lat": lat_coord, "lon": lon_coord}

    return combined, coords

    # print("dataGP: ", dataGP)
    # print("dataT: ", dataT)

    # zData = dataGP["z"].values
    # print(zData)
    # print("\n")
    # tData = dataT["t2m"].values
    # print(tData)

    # print("Times match: ", np.array_equal(dataGP.time.values, dataT.time.values))


def split_data_by_years(data, coords, train_end=2015, val_end=2017):
    """Split by year ranges"""
    times = pd.DatetimeIndex(coords["time"])

    train_mask = times.year <= train_end
    val_mask = (times.year > train_end) & (times.year <= val_end)
    test_mask = times.year > val_end

    train_data = data[np.where(train_mask)[0]]
    val_data = data[np.where(val_mask)[0]]
    test_data = data[np.where(test_mask)[0]]

    # print(f"\nData split:")
    # print(f"  Train: 1979-{train_end} → {train_data.shape[0]:,} timesteps")
    # print(f"  Val: {train_end+1}-{val_end} → {val_data.shape[0]:,} timesteps")
    # print(f"  Test: {val_end+1}-... → {test_data.shape[0]:,} timesteps")

    return train_data, val_data, test_data


def normalize_data_stats(data, stats=None):
    # normalize data to zero mean, unit variance for each variable
    # data: [time, lat, lon, vars]

    # print(f"Data received - shape: {data.shape}")
    # print(f"Data received - GP mean: {data[:,:,:,0].mean():.2f}")
    # print(f"Data received - T mean: {data[:,:,:,1].mean():.2f}")

    if stats == None:
        # print("Testing different axes:")
        # print(f"axis=0 shape: {np.mean(data, axis=0).shape}")
        # print(f"axis=(0,1,2) shape: {np.mean(data, axis=(0, 1, 2)).shape}")
        # print(f"axis=(0,1,2) keepdims shape: {np.mean(data, axis=(0, 1, 2), keepdims=True).shape}")

        mean_gp = data[:, :, :, 0].mean()
        mean_t = data[:, :, :, 1].mean()
        mean = np.array([[[[mean_gp, mean_t]]]]).astype(data.dtype)

        std_gp = data[:, :, :, 0].std()
        std_t = data[:, :, :, 1].std()
        std = np.array([[[[std_gp, std_t]]]]).astype(data.dtype)

        # print(f"Data shape: {data.shape}")
        # print(f"Data type: {data.dtype}")
        # print(f"Data min: {data.min()}")
        # print(f"Data max: {data.max()}")
        # print(f"Mean shape: {mean.shape}")
        # print(f"Mean values: {mean.squeeze()}")
        # print(f"Std shape: {std.shape}")
        # print(f"Std values: {std.squeeze()}")

        stats = {"mean": mean, "std": std}
        # print("Computed new normalization stats for this data")
    else:
        mean = stats["mean"]
        std = stats["std"]
        # print("using provided normalization stats")

    normalized = (data - mean) / (std + 1e-6)

    return normalized, stats


def create_sequences(data, num_ar_steps=4):
    # create autoregressive sequence
    # data: [time, lat, lon, num_vars]
    # num_ar_steps

    # Calculate how many sequences we can create
    num_sequences = data.shape[0] - num_ar_steps
    # print(f"\nCreating sequences:")
    # print(f"  Total timesteps: {data.shape[0]}")
    # print(f"  Autoregressive steps: {num_ar_steps}")
    # print(f"  Number of sequences: {num_sequences}")

    # Inputs are the starting states
    inputs = data[:num_sequences, :, :, :]

    # Targets are the next num_ar_steps timesteps for each input
    targets = []
    for i in range(num_sequences):
        # Get timesteps [i+1, i+2, ..., i+num_ar_steps]
        target_sequence = data[i + 1 : i + 1 + num_ar_steps, :, :, :]
        targets.append(target_sequence)

    targets = np.array(targets)

    # print(f"  Input shape: {inputs.shape}")
    # print(f"  Target shape: {targets.shape}")

    return inputs, targets


if __name__ == "__main__":
    # Load all data
    data, coords = load_combined_data()

    # CHECK RAW DATA IMMEDIATELY
    # print("RAW DATA CHECK (BEFORE SPLIT)")
    # print(f"Combined data shape: {data.shape}")
    # print(f"Geopotential [var 0]:")
    # print(f"  Min: {data[:,:,:,0].min():.2f}")
    # print(f"  Max: {data[:,:,:,0].max():.2f}")
    # print(f"  Mean: {data[:,:,:,0].mean():.2f}")
    # print(f"Temperature [var 1]:")
    # print(f"  Min: {data[:,:,:,1].min():.2f}")
    # print(f"  Max: {data[:,:,:,1].max():.2f}")
    # print(f"  Mean: {data[:,:,:,1].mean():.2f}")

    print("Data Loading Complete")

    # Split
    train, val, test = split_data_by_years(data, coords)

    # print("TRAIN DATA IMMEDIATELY AFTER SPLIT")
    # print(f"Train shape: {train.shape}")
    # print(f"Train GP [var 0] mean: {train[:,:,:,0].mean():.2f}")
    # print(f"Train T [var 1] mean: {train[:,:,:,1].mean():.2f}")
    # print(f"Train GP min/max: {train[:,:,:,0].min():.2f} / {train[:,:,:,0].max():.2f}")
    # print(f"Train T min/max: {train[:,:,:,1].min():.2f} / {train[:,:,:,1].max():.2f}")

    print(f"\nReady for Normalization!")

    # normalize train and get stats
    train_norm, stats = normalize_data_stats(train, stats=None)

    # normalize valid and test on train
    val_norm, _ = normalize_data_stats(val, stats=stats)
    test_norm, _ = normalize_data_stats(test, stats=stats)

    print("FINAL DATA SHAPES")
    print(f"Train: {train_norm.shape}")
    print(f"Val: {val_norm.shape}")
    print(f"Test: {test_norm.shape}")

    # FREE MEMORY
    del data, train, val, test
    import gc

    gc.collect()
    print("\nFreed original data from memory")

    # SUBSAMPLE (optional - remove this later when you have more RAM)
    print("\nSubsampling data to reduce memory usage...")
    train_norm = train_norm[::4]  # Keep 25% of timesteps
    val_norm = val_norm[::4]
    test_norm = test_norm[::4]
    print(f"Train after subsample: {train_norm.shape[0]} timesteps")

    # Now create sequences
    print("\nCreating sequences...")
    train_inputs, train_targets = create_sequences(train_norm, num_ar_steps=4)
    val_inputs, val_targets = create_sequences(val_norm, num_ar_steps=4)
    test_inputs, test_targets = create_sequences(test_norm, num_ar_steps=4)

    print("\nSaving preprocessed data...")
    np.savez_compressed(
        "preprocessed_data.npz",
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        stats=stats,
    )

    print("Success!")
