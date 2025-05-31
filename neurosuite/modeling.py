from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut
import numpy as np

class EEGModel:
    def __init__(self, model_name):
        self.model = self._get_model(model_name)

    def _get_model(self, model_name):
        if model_name == "svm":
            return SVC(kernel="rbf", C=1, probability=True)
        elif model_name == "rf":
            return RandomForestClassifier(n_estimators=100)
        elif model_name == "xgb":
            return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def train(self, X, y, groups=None, cross_subject=False):
        if cross_subject and groups is not None:
            logo = LeaveOneGroupOut()
            scores = cross_val_score(self.model, X, y, cv=logo, groups=groups)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X, y, cv=cv)
        return {
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
            "cv_scores": scores
        }
