import numpy as np
from scipy.signal import butter, filtfilt
import mne

class Preprocessor:
    def __init__(self, config):
        self.fs = config.get("sampling_rate", 250)
        self.lowcut = config.get("lowcut", 1)
        self.highcut = config.get("highcut", 40)
        self.apply_filter = config.get("apply_filter", True)
        self.apply_car = config.get("apply_car", False)
        self.apply_ica = config.get("apply_ica", False)
        self.apply_baseline = config.get("apply_baseline", False)
        self.baseline_window = config.get("baseline_window", (0, 1)) 

        # Channel names and types (you can generalize this later)
        self.ch_names = config.get("channel_names", None)
        self.ch_types = config.get("channel_types", "eeg")

    def bandpass_filter(self, data):
        b, a = butter(4, [self.lowcut / (0.5 * self.fs), self.highcut / (0.5 * self.fs)], btype="band")
        return filtfilt(b, a, data, axis=0)

    def common_average_reference(self, data):
        avg = np.mean(data, axis=0, keepdims=True)
        return data - avg

    def baseline_correction(self, data):
        start, end = self.baseline_window
        start_idx, end_idx = int(start * self.fs), int(end * self.fs)
        baseline = data[:, start_idx:end_idx].mean(axis=1, keepdims=True)
        return data - baseline

    def apply_ica_mne(self, data):
        if self.ch_names is None:
            n_channels = data.shape[0]
            self.ch_names = [f"EEG {i+1}" for i in range(n_channels)]

        info = mne.create_info(ch_names=self.ch_names, sfreq=self.fs, ch_types=self.ch_types)
        raw = mne.io.RawArray(data, info)
        raw.filter(self.lowcut, self.highcut, fir_design='firwin', verbose=False)

        ica = mne.preprocessing.ICA(n_components=0.95, random_state=42, max_iter="auto", verbose=False)
        ica.fit(raw)
        ica.detect_artifacts(raw)
        raw_clean = ica.apply(raw.copy())
        return raw_clean.get_data()

    def transform(self, X):
        X_out = []
        for trial in X:
            if self.apply_ica:
                trial = self.apply_ica_mne(trial)

            if self.apply_filter:
                trial = np.array([self.bandpass_filter(ch) for ch in trial])

            if self.apply_car:
                trial = self.common_average_reference(trial)

            if self.apply_baseline:
                trial = self.baseline_correction(trial)

            X_out.append(trial)
        return np.array(X_out)
