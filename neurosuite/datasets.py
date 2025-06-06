import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_dataset(name):
    name = name.lower()
    if name == "deap":
        return load_generic("deap", "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/")
    elif name == "seed":
        return load_generic("seed", "https://bcmi.sjtu.edu.cn/~seed/index.html")
    elif name == "openneuro":
        return load_generic("openneuro", "https://openneuro.org/")
    elif name == "custom single-file":
        return load_custom_single()
    elif name == "custom multi-file":
        return load_custom_multi()
    else:
        raise ValueError(f"Unknown dataset name: {name}")

def load_generic(dataset_name, download_url):
    npz_path = f"datasets/{dataset_name}_data.npz"
    mat_path = f"datasets/{dataset_name}_data.mat"

    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        return data["X"], np.array(data["y"]).flatten(), np.array(data["groups"]).flatten()

    elif os.path.exists(mat_path):
        mat = loadmat(mat_path)
        X = mat.get("X") or mat.get("data")
        if X is None:
            raise ValueError(f" 'X' data not found in {mat_path}.")
        y = mat.get("y") or mat.get("labels") or np.zeros(X.shape[0])
        groups = mat.get("groups") or mat.get("subjs") or np.zeros(X.shape[0])
        return X, np.array(y).flatten(), np.array(groups).flatten()

    else:
        raise FileNotFoundError(
            f"{dataset_name.upper()} dataset not found.\n"
            f"Please download it from: {download_url}\n"
            f"Then place one of the following files in the 'datasets/' folder:\n"
            f"📁 datasets/{dataset_name}_data.npz (preferred NumPy format)\n"
            f"📁 OR datasets/{dataset_name}_data.mat (original MATLAB format)"
        )

def load_custom_single(uploaded_file=None):
    if uploaded_file is None:
        raise FileNotFoundError("No uploaded custom single-file found.")

    if uploaded_file.name.endswith(".npz"):
        data = np.load(uploaded_file, allow_pickle=True)
        return data["X"], np.array(data["y"]).flatten(), np.array(data["groups"]).flatten()

    elif uploaded_file.name.endswith(".mat"):
        mat = loadmat(uploaded_file)
        X = mat.get("X") or mat.get("data")
        if X is None:
            raise ValueError(" 'X' data not found in uploaded .mat file.")
        y = mat.get("y") or mat.get("labels") or np.zeros(X.shape[0])
        groups = mat.get("groups") or mat.get("subjs") or np.zeros(X.shape[0])
        return X, np.array(y).flatten(), np.array(groups).flatten()

    else:
        raise ValueError("Unsupported file type. Please upload a .npz or .mat file.")

def load_custom_multi(files=None, meta_file=None):
    if files is None or len(files) == 0:
        raise FileNotFoundError("No EEG .mat files provided for multi-subject custom dataset.")

    X_all, y_all, group_all = [], [], []
    meta = None

    if meta_file is not None:
        meta = pd.read_csv(meta_file)

    for file in files:
        filename = file.name
        mat = loadmat(file)

        if "X_event" not in mat:
            continue 

        data = np.transpose(mat["X_event"], (2, 0, 1)) 
        X_all.append(data)

        if meta is not None and filename in meta["filename"].values:
            label = meta.loc[meta["filename"] == filename, "y"].values[0]
            y_all.append(np.full(data.shape[0], label))
        else:
            y_all.append(np.full(data.shape[0], -1)) 

        subject_id = filename.split("_")[0]
        group_all.append(np.full(data.shape[0], subject_id, dtype=object))

    return (
        np.concatenate(X_all),
        np.concatenate(y_all),
        np.concatenate(group_all)
    )

