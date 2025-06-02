import os
import numpy as np
from scipy.io import loadmat

def load_dataset(name):
    name = name.lower()
    if name == "deap":
        return load_generic("deap", "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/")
    elif name == "seed":
        return load_generic("seed", "https://bcmi.sjtu.edu.cn/~seed/index.html")
    elif name == "openneuro":
        return load_generic("openneuro", "https://openneuro.org/")
    elif name == "custom":
        raise ValueError("Custom upload should be handled separately via `load_custom_file()`.")
    else:
        raise ValueError(f"Unknown dataset name: {name}")

def load_generic(dataset_name, download_url):
    npz_path = f"datasets/{dataset_name}_data.npz"
    mat_path = f"datasets/{dataset_name}_data.mat"

    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        return data["X"], data["y"], data["groups"]

    elif os.path.exists(mat_path):
        mat = loadmat(mat_path)

        # Fallback keys depending on the dataset structure
        X = mat.get("X") or mat.get("data")
        y = mat.get("y") or mat.get("labels") or np.zeros(X.shape[0])
        groups = mat.get("groups") or mat.get("subjs") or np.zeros(X.shape[0])

        return X, y, groups

    else:
        raise FileNotFoundError(
            f"❌ {dataset_name.upper()} dataset not found.\n"
            f"👉 Please download it from: {download_url}\n"
            f"Then place one of the following files in the 'datasets/' folder:\n"
            f"📁 datasets/{dataset_name}_data.npz (preferred NumPy format)\n"
            f"📁 OR datasets/{dataset_name}_data.mat (original MATLAB format)"
        )

def load_custom_file(uploaded_file):
    if uploaded_file.name.endswith(".npz"):
        data = np.load(uploaded_file, allow_pickle=True)
        return data["X"], data["y"], data["groups"]

    elif uploaded_file.name.endswith(".mat"):
        mat = loadmat(uploaded_file)

        X = mat.get("X") or mat.get("data")
        y = mat.get("y") or mat.get("labels") or np.zeros(X.shape[0])
        groups = mat.get("groups") or mat.get("subjs") or np.zeros(X.shape[0])

        return X, y, groups

    else:
        raise ValueError("❌ Unsupported file type. Please upload a .npz or .mat file.")

import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_multisubject_odors(files, meta_file=None):
    X_all, y_all, group_all = [], [], []
    meta = None

    if meta_file is not None:
        meta = pd.read_csv(meta_file)

    for file in files:
        filename = file.name
        mat = loadmat(file)

        if "X_event" not in mat:
            continue  # Skip if no EEG data

        data = np.transpose(mat["X_event"], (2, 0, 1))  # (N, 250, 216)
        X_all.append(data)

        # Assign label (y)
        if meta is not None and filename in meta["filename"].values:
            label = meta.loc[meta["filename"] == filename, "y"].values[0]
            y_all.append(np.full(data.shape[0], label))
        else:
            y_all.append(np.full(data.shape[0], -1))  # -1 means unlabeled

        # Assign group (subject) from filename prefix
        subject_id = filename.split("_")[0]  # e.g., 'Subject1'
        group_all.append(np.full(data.shape[0], subject_id))

    return (
        np.concatenate(X_all),
        np.concatenate(y_all),
        np.concatenate(group_all),
    )

