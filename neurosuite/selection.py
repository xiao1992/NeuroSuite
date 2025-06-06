import numpy as np
from sklearn.feature_selection import f_classif

class ElectrodeSelector:
    def __init__(self, ch_names):
        self.ch_names = ch_names

    def rank_by_fdr(self, X, y, band_split=5):
        f_scores, _ = f_classif(X, y)

        if band_split > 1:
            f_scores_per_ch = np.array_split(f_scores, len(f_scores) // band_split)
            avg_scores = np.array([np.mean(fs) for fs in f_scores_per_ch])
        else:
            avg_scores = f_scores

        ch_ranking = sorted(
            zip(self.ch_names, avg_scores), key=lambda x: x[1], reverse=True
        )
        return ch_ranking
