import shap
import numpy as np
import matplotlib.pyplot as plt
import mne

def explain_model(model, X, method="tree"):
    explainer = shap.Explainer(model, X) if method != "tree" else shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values

def plot_shap_topomap(shap_values, ch_names, info=None, title="SHAP Topomap"):
    mean_vals = np.mean(np.abs(shap_values.values), axis=0)

    if mean_vals.shape[0] > len(ch_names):
        ch_scores = np.array_split(mean_vals, len(ch_names))
        ch_scores = [np.mean(bandvals) for bandvals in ch_scores]
    else:
        ch_scores = mean_vals

    if info is None:
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types='eeg')
        info.set_montage(montage)

    evoked = mne.EvokedArray(np.expand_dims(ch_scores, axis=1), info)
    fig = evoked.plot_topomap(times=[0], size=3, scalings=1, time_format="", colorbar=True)
    plt.suptitle(title)
    return fig
