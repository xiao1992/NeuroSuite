import matplotlib.pyplot as plt
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap

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

def plot_topomap_class_difference(X, y, class_a, class_b, compute_band_power, bands, sfreq=250, ch_names=None, title_prefix="Topomap", vlim=0.03):
    if ch_names is None:
        montage = make_standard_montage('GSN-HydroCel-256')
        ch_names = montage.ch_names[:X.shape[1]]
    else:
        montage = make_standard_montage('GSN-HydroCel-256')

    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)

    class_a_idx = np.where(np.array(y) == class_a)[0]
    class_b_idx = np.where(np.array(y) == class_b)[0]

    if len(class_a_idx) == 0 or len(class_b_idx) == 0:
        raise ValueError("Selected classes are not found in the provided labels.")

    X_a = X[class_a_idx]
    X_b = X[class_b_idx]

    band_a = compute_band_power(X_a)
    band_b = compute_band_power(X_b)

    fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 4))
    if len(bands) == 1:
        axes = [axes]

    for i, band in enumerate(bands):
        mean_a = np.mean(band_a[band], axis=1)
        mean_b = np.mean(band_b[band], axis=1)
        diff = mean_a - mean_b

        im, _ = plot_topomap(
            diff,
            pos=info,
            axes=axes[i],
            show=False,
            cmap='RdBu_r',
            vlim=(-vlim, vlim),
            contours=6,
            sphere=0.07,
        )
        axes[i].set_title(band.capitalize())

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label("Power Difference (A - B)")
    fig.suptitle(f"{title_prefix}: {class_a} vs {class_b}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

