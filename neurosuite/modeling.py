from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, f1_score

class EEGModel:
    def __init__(self, model_name):
        self.model = self._get_model(model_name)

    def _get_model(self, model_name):
        if model_name == "svm":
            return make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=1, probability=True, random_state=42)
            )
        elif model_name == "rf":
            return make_pipeline(
                StandardScaler(),
                RandomForestClassifier(n_estimators=100, random_state=42)
            )
        elif model_name == "xgb":
            return make_pipeline(
                StandardScaler(),
                XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            )
            else:
                raise ValueError(f"Unsupported model: {model_name}")

    def train(self, X, y, groups=None, cross_subject=False):
        if cross_subject and groups is not None:
            cv = LeaveOneGroupOut()
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
        acc_scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy", groups=groups)
        f1_scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1_weighted", groups=groups)
    
        return {
            "mean_accuracy": np.mean(acc_scores),
            "std_accuracy": np.std(acc_scores),
            "mean_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores),
            "cv_scores": acc_scores.tolist()  # or f1_scores if preferred in chart
        }
