# neurosuite/__init__.py

__version__ = "0.1.0"

from .pipeline import EEGPipeline
from .features import EEGFeatures
from .modeling import EEGModel
from .interpretation import EEGExplainer
from .graphs import EEGTemporalGraph
from .visualization import plot_shap_topomap
from .datasets import load_dataset
from .selection import ElectrodeSelector
from .domain import CORAL
from .ui import run_gui
