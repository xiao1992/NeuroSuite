from neurosuite.datasets import load_dataset
from neurosuite.preprocessing import Preprocessor
from neurosuite.features import EEGFeatures
from neurosuite.modeling import EEGModel
from neurosuite.domain import CORAL

class EEGPipeline:
    def __init__(self, config, cross_subject=False):
        self.config = config
        self.cross_subject = cross_subject
        self.X, self.y, self.groups = None, None, None
        self.features = None
        self.model = None
        self.results = None
        
    def set_data(self, X, y, groups=None):
        self.X = X
        self.y = y
        self.groups = groups if groups is not None else np.zeros(len(y))
        return self

    def load_data(self):
        self.X, self.y, self.groups = load_dataset(self.config["dataset"])
        return self

    def preprocess(self):
        pre = Preprocessor(self.config)
        self.X = pre.transform(self.X)
        return self
    
    def extract_features(self):
        fe = EEGFeatures(self.config)
        self.features = fe.transform(self.X)
        return self

    def adapt(self):
        if self.config.get("use_coral"):
            self.features = CORAL().fit_transform(self.features, self.groups)
        return self
    
    def fit(self):
        name_map = {
            "svm": "svm",
            "randomforest": "rf",
            "randomforestclassifier": "rf",
            "rf": "rf",
            "xgboost": "xgb",
            "xgb": "xgb",
            "xgboostclassifier": "xgb"
        }
        model_key = self.config["model"].lower().replace(" ", "")
        mapped_model = name_map.get(model_key)
    
        if mapped_model is None:
            raise ValueError(f"Unsupported model: {self.config['model']}")
    
        self.model = EEGModel(mapped_model)
        self.results = self.model.train(self.features, self.y, self.groups, self.cross_subject)
        return self

    def evaluate(self):
        return self.results

    def run_all(self):
        return self.load_data().preprocess().extract_features().adapt().fit().evaluate()
