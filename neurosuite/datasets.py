import os
import numpy as np

def load_dataset(name):
    if name.lower() == "deap":
        return load_deap()
    elif name.lower() == "seed":
        return load_seed()
    elif name.lower() == "openneuro":
        return load_openneuro()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def load_deap():
    file_path = "datasets/deap_data.npz"
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "❌ DEAP dataset not found.\n"
            "👉 Please download it from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/\n"
            "and place the file as: datasets/deap_data.npz"
        )
    data = np.load(file_path, allow_pickle=True)
    return data["X"], data["y"], data["groups"]

def load_seed():
    file_path = "datasets/seed_data.npz"
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "❌ SEED dataset not found.\n"
            "👉 Please download it from https://bcmi.sjtu.edu.cn/~seed/index.html\n"
            "and place the file as: datasets/seed_data.npz"
        )
    data = np.load(file_path, allow_pickle=True)
    return data["X"], data["y"], data["groups"]

def load_openneuro():
    file_path = "datasets/openneuro_data.npz"
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "❌ OpenNeuro dataset not found.\n"
            "👉 Please visit https://openneuro.org/ to select a dataset, download it,\n"
            "and place the file as: datasets/openneuro_data.npz"
        )
    data = np.load(file_path, allow_pickle=True)
    return data["X"], data["y"], data["groups"]
