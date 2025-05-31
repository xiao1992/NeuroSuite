# neurosuite/features.py

import numpy as np
from scipy.signal import welch

class EEGFeatures:
    def __init__(self, config):
        self.config = config
        self.fs = 128  # sampling rate

    def transform(self, X):
        # Input X: shape (samples, time, channels)
        all_features = []
        for trial in X:
            trial_features = []
            for ch in trial.T:  # loop over channels
                freqs, psd = welch(ch, fs=self.fs, nperseg=self.fs)
                trial_features.extend(self.band_power(psd, freqs))
            all_features.append(trial_features)
        return np.array(all_features)

    def band_power(self, psd, freqs):
        # Define bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        features = []
        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.sum(psd[idx]))
        return features
