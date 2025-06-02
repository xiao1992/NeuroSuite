import streamlit as st
from neurosuite import EEGPipeline
from neurosuite.datasets import load_dataset, load_custom_file, load_multisubject_odors

st.set_page_config(page_title="NeuroSuite GUI", layout="wide")
st.title("🧠 NeuroSuite: EEG Processing & Modeling")

# Sidebar options
st.sidebar.header("Configuration")
dataset = st.sidebar.selectbox("Select Dataset", ["DEAP", "SEED", "OpenNeuro", "Custom Single File", "Custom Multi-File"])
model = st.sidebar.selectbox("Select Model", ["SVM", "RandomForest", "XGBoost"])
cross_subject = st.sidebar.checkbox("Cross-subject evaluation", value=True)
use_coral = st.sidebar.checkbox("Use CORAL adaptation", value=False)

# Upload blocks
uploaded_file = None
custom_files = []
meta_file = None
custom_key_map = {}

if dataset == "Custom Single File":
    uploaded_file = st.sidebar.file_uploader("Upload .npz or .mat file", type=["npz", "mat"])

    if uploaded_file is not None:
        import tempfile
        import numpy as np
        from scipy.io import loadmat

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if uploaded_file.name.endswith(".npz"):
            keys = list(np.load(tmp_path, allow_pickle=True).keys())
        else:
            keys = [k for k in loadmat(tmp_path) if not k.startswith("__")]

        custom_key_map["X"] = st.selectbox("Select EEG data key", keys)
        y_key = st.selectbox("Select label key (optional)", ["None"] + keys)
        g_key = st.selectbox("Select group key (optional)", ["None"] + keys)

        if y_key != "None":
            custom_key_map["y"] = y_key
        if g_key != "None":
            custom_key_map["groups"] = g_key

elif dataset == "Custom Multi-File":
    custom_files = st.sidebar.file_uploader("Upload multiple .mat files", type="mat", accept_multiple_files=True)
    meta_file = st.sidebar.file_uploader("Upload metadata CSV (optional)", type="csv")

# Run Pipeline
if st.sidebar.button("Run Pipeline"):
    config = {"dataset": dataset, "model": model, "use_coral": use_coral}
    st.info("🔄 Loading and processing data...")

    try:
        if dataset == "Custom Single File":
            X, y, groups = load_custom_file(uploaded_file, custom_key_map)
        elif dataset == "Custom Multi-File":
            X, y, groups = load_multisubject_odors(custom_files, meta_file)
        else:
            X, y, groups = load_dataset(dataset)

        pipeline = EEGPipeline(config=config, cross_subject=cross_subject)
        pipeline.set_data(X, y, groups)
        results = pipeline.preprocess().extract_features().adapt().fit().evaluate()

        st.success("✅ Pipeline completed successfully!")
        st.metric("Mean", f"{results['mean_accuracy']:.3f}")
        st.metric("Std", f"{results['std_accuracy']:.3f}")
        st.line_chart(results["cv_scores"])

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")
