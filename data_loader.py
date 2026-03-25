import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

# Target houses and appliances
TARGET_HOUSES = [1, 2, 5]
TARGET_APPLIANCES = ['fridge', 'dishwasher', 'microwave', 'washer_dryer', 'kettle']

# Known meter-to-appliance mapping for UKDALE houses 1, 2, 5
# meter 1 is always the aggregate mains
HOUSE_METER_MAP = {
    1: {
        'kettle':      2,
        'microwave':   3,
        'dishwasher':  4,
        'fridge':      5,
        'washer_dryer': 6,
    },
    2: {
        'kettle':      2,
        'dishwasher':  3,
        'fridge':      4,
        'microwave':   5,
        'washer_dryer': 8,
    },
    5: {
        'kettle':      2,
        'microwave':   4,
        'dishwasher':  5,
        'fridge':      6,
        'washer_dryer': 8,
    },
}


class UKDaleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def read_meter(h5_path, building, meter):
    """
    Read power data for a single meter from the UKDALE HDF5 file.
    Returns a pandas Series with a DatetimeIndex.
    """
    key = f'/building{building}/elec/meter{meter}/table'
    df = pd.read_hdf(h5_path, key=key)
    timestamps = pd.to_datetime(df['index'], unit='ns', utc=True)
    return pd.Series(df['values_block_0'].values.astype(np.float32), index=timestamps)


def align_and_resample(mains_series, appliance_series, resample_freq='8s'):
    """
    Resample both series to a uniform frequency and align on timestamps.
    Fills gaps with forward-fill then zero.
    """
    mains = mains_series.resample(resample_freq).mean().ffill().fillna(0)
    appliance = appliance_series.resample(resample_freq).mean().ffill().fillna(0)

    # Align on common timestamps
    df = pd.DataFrame({'mains': mains, 'appliance': appliance}).dropna()
    return df['mains'].values.astype(np.float32), df['appliance'].values.astype(np.float32)


def create_sequences(mains, appliance, window_size, target_size=1):
    """
    Create sliding window sequences (sequence-to-point by default).
    """
    X, y = [], []
    stride = 5

    for i in range(0, len(mains) - window_size - target_size + 1, stride):
        X.append(mains[i:i + window_size])
        if target_size == 1:
            midpoint = i + window_size // 2
            y.append(appliance[midpoint:midpoint + 1])
        else:
            y.append(appliance[i + window_size:i + window_size + target_size])

    return np.array(X).reshape(-1, window_size, 1), np.array(y)


def load_house(h5_path, building, window_size=100, target_size=1,
               train_ratio=0.7, val_ratio=0.15, normalize=True):
    """
    Load and preprocess all target appliances for one house.

    Returns:
        dict mapping appliance_name -> data dict with DataLoaders and metadata
    """
    print(f"\n=== House {building} ===")
    meter_map = HOUSE_METER_MAP.get(building, {})

    # Load aggregate mains (meter 1)
    mains_series = read_meter(h5_path, building, meter=1)
    print(f"  Mains: {len(mains_series)} samples  "
          f"({mains_series.index[0]} -> {mains_series.index[-1]})")

    results = {}

    for appliance in TARGET_APPLIANCES:
        meter = meter_map.get(appliance)
        if meter is None:
            print(f"  [SKIP] '{appliance}' — no meter mapping for house {building}")
            continue

        try:
            appliance_series = read_meter(h5_path, building, meter)
        except KeyError:
            print(f"  [SKIP] '{appliance}' — meter {meter} not found in file")
            continue

        print(f"  [FOUND] '{appliance}' at meter {meter}: {len(appliance_series)} samples")

        # Resample and align
        mains_data, appliance_data = align_and_resample(mains_series, appliance_series)

        # Fit scalers on training portion only to avoid data leakage
        train_end_idx = int(train_ratio * len(mains_data))
        if normalize:
            mains_scaler = StandardScaler()
            appliance_scaler = StandardScaler()
            mains_scaler.fit(mains_data[:train_end_idx].reshape(-1, 1))
            appliance_scaler.fit(appliance_data[:train_end_idx].reshape(-1, 1))
            mains_norm = mains_scaler.transform(mains_data.reshape(-1, 1)).flatten()
            appliance_norm = appliance_scaler.transform(appliance_data.reshape(-1, 1)).flatten()
        else:
            mains_norm = mains_data
            appliance_norm = appliance_data
            mains_scaler = appliance_scaler = None

        # Create sequences
        X, y = create_sequences(mains_norm, appliance_norm, window_size, target_size)

        # Sequential split (preserves temporal order)
        n = len(X)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        X_train, y_train = X[:train_end],        y[:train_end]
        X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
        X_test,  y_test  = X[val_end:],          y[val_end:]

        print(f"    Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

        train_loader = DataLoader(UKDaleDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader   = DataLoader(UKDaleDataset(X_val,   y_val),   batch_size=32, shuffle=False)
        test_loader  = DataLoader(UKDaleDataset(X_test,  y_test),  batch_size=32, shuffle=False)

        results[appliance] = {
            'train_loader':      train_loader,
            'val_loader':        val_loader,
            'test_loader':       test_loader,
            'mains_scaler':      mains_scaler,
            'appliance_scaler':  appliance_scaler,
            'appliance_name':    appliance,
            'window_size':       window_size,
            'target_size':       target_size,
            'input_size':        1,
            'output_size':       target_size,
        }

    return results


def load_all_houses(h5_path, window_size=100, target_size=1,
                    train_ratio=0.7, val_ratio=0.15, normalize=True):
    """
    Load and preprocess houses 1, 2, and 5 for all target appliances.

    Returns:
        dict: { house_id (int) -> { appliance_name -> data dict } }
    """
    all_data = {}
    for building in TARGET_HOUSES:
        all_data[building] = load_house(
            h5_path, building, window_size, target_size,
            train_ratio, val_ratio, normalize
        )
    return all_data


def explore_h5_structure(h5_path):
    """
    Print the meter structure of the UKDALE HDF5 file.
    Useful for verifying meter numbers before loading.
    """
    import h5py
    print(f"Exploring: {h5_path}\n")
    with h5py.File(h5_path, 'r') as f:
        for building in TARGET_HOUSES:
            key = f'building{building}/elec'
            if key not in f:
                print(f"  House {building}: not found")
                continue
            meters = sorted(f[key].keys())
            print(f"  House {building}: {meters}")


# ---------------------------------------------------------------------------
# Compatibility wrappers — used by train_*.py scripts
# ---------------------------------------------------------------------------
H5_PATH = "preprocessed_datasets/ukdale/ukdale.h5"


def explore_available_appliances(_file_path):
    """
    Compat wrapper: returns {index: appliance_name} for TARGET_APPLIANCES.
    _file_path is ignored (kept for API compatibility with train_*.py).
    """
    return {i: name for i, name in enumerate(TARGET_APPLIANCES)}


def load_and_preprocess_ukdale(file_path, appliance_index, window_size=100,
                                target_size=1, train_ratio=0.7, val_ratio=0.15,
                                normalize=True):
    """
    Compat wrapper: maps (file_path, appliance_index) to the new H5 loader.
    Extracts house number from file_path (e.g. 'ukdale1.mat' -> house 1).
    """
    import re
    match = re.search(r'ukdale(\d+)', os.path.basename(file_path))
    building = int(match.group(1)) if match else 1

    appliance_name = TARGET_APPLIANCES[appliance_index]
    house_data = load_house(H5_PATH, building, window_size, target_size,
                            train_ratio, val_ratio, normalize)

    if appliance_name not in house_data:
        raise ValueError(f"'{appliance_name}' not found in house {building}")

    return house_data[appliance_name]


if __name__ == "__main__":
    h5_path = "preprocessed_datasets/ukdale/ukdale.h5"

    # Explore structure first
    explore_h5_structure(h5_path)

    # Load all houses and appliances
    all_data = load_all_houses(h5_path, window_size=100, target_size=1)

    print("\n=== Summary ===")
    for house, appliances in all_data.items():
        print(f"House {house}:")
        for appliance, d in appliances.items():
            print(f"  {appliance}: "
                  f"train={len(d['train_loader'])} batches, "
                  f"val={len(d['val_loader'])} batches, "
                  f"test={len(d['test_loader'])} batches")
