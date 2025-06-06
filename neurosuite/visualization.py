import matplotlib.pyplot as plt
import numpy as np

def plot_trial(trial, ch_names=None):
    time = np.arange(trial.shape[0])
    plt.figure(figsize=(10, 6))
    for i in range(trial.shape[1]):
        plt.plot(time, trial[:, i] + i * 50, label=ch_names[i] if ch_names else f"Ch {i}")
    plt.xlabel("Time")
    plt.title("EEG Trial")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
