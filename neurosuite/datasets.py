
import os
import numpy as np
import scipy.io
import zipfile
import tempfile
from sklearn.preprocessing import StandardScaler
import mne
import requests
import io

def load_dataset(name):
    if name.lower() == "deap":
        return load_deap()
    elif name.lower() == "seed":
        return load_seed()
    elif name.lower() == "openneuro":
        return load_openneuro()
    else:
        raise ValueError(f"Unsupported dataset: {name}")

# -----------------------------------------
# DEAP Dataset
# -----------------------------------------

def load_deap():
    url = "https://www.dropbox.com/s/7a2w9jtu7c56z57/deap_data.npz?dl=1"
    file_path = fetch_file(url, "deap_data.npz")
    data = np.load(file_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    return X, y, groups

# -----------------------------------------
# SEED Dataset
# -----------------------------------------

def load_seed():
    url = "https://www.dropbox.com/s/9r9tm2tcqseikpc/seed_data.npz?dl=1"
    file_path = fetch_file(url, "seed_data.npz")
    data = np.load(file_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    return X, y, groups

# -----------------------------------------
# OpenNeuro Dataset (Example: ds002778)
# -----------------------------------------

def load_openneuro():
    url = "https://www.dropbox.com/s/3f7s9sxy2trdn58/openneuro_data.npz?dl=1"
    file_path = fetch_file(url, "openneuro_data.npz")
    data = np.load(file_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    return X, y, groups

# -----------------------------------------
# Download + Cache Utility
# -----------------------------------------

def fetch_file(url, filename, cache_dir="datasets"):
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_path):
        print(f"📥 Downloading {filename}...")
        r = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path
