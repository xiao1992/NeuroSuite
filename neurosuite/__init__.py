__version__ = "0.1.0"

from .pipeline import EEGPipeline
from .features import EEGFeatures
from .modeling import EEGModel
from .datasets import load_dataset
from .domain import CORAL
from .interpretation import explain_model, plot_shap_topomap
from .selection import ElectrodeSelector
from .visualization import plot_trial
