# NeuroSuite
Interactive lab-like toolkit for EEG prep and modeling

It is an open-source Python toolkit for EEG modeling, built for both cognitive neuroscience researchers and machine learning engineers. It provides plug-and-play pipelines for EEG data fetching (DEAP, SEED, OpenNeuro), preprocessing, feature extraction, model training, interpretation, and interactive exploration — all within a unified GUI beginner-friendly framework.

---

## Key Features
- Load Popular EEG Datasets (DEAP, SEED, OpenNeuro) with one click
- Preprocess EEG Signals with built-in filters and epoching including Bandpass filtering (1–40 Hz), Epoching into fixed-length trials (e.g. 2s), Baseline removal, Trial reshaping (samples × time × channels), Group tracking (for cross-subject validation), Notch filtering (50/60 Hz), ICA for artifact removal, Common average referencing, Z-score normalization
- Extract Features like power spectral density, workload index, entropy, etc.
- Apply Machine Learning Models (SVM, XGBoost, Random Forest)
- Cross-Subject Generalization with optional CORAL domain adaptation
- SHAP Interpretation with visual topomap overlays
- Electrode Ranking and feature contribution analysis
- Temporal Graph Construction from EEG epochs
- GUI Interface for no-code usage

---

## Installation
Development mode:
```bash
pip install -e .

Direct GitHub install (non-editable):
```bash
pip install git+https://github.com/your-username/NeuroSuite.git
pip install -e .
