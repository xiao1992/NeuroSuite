import streamlit as st
from neurosuite.datasets import load_dataset, load_custom_file
from neurosuite.pipeline import EEGPipeline

st.set_page_config(page_title="NeuroSuite", layout="centered")
st.title("NeuroSuite: EEG Modeling Toolkit")

# Sidebar configuration
st.sidebar.title("⚙️ Settings")

dataset = st.sidebar.selectbox("Select Dataset", ["DEAP", "SEED", "OpenNeuro", "Custom"])
model = st.sidebar.selectbox("Select Model", ["XGBoost", "RandomForest", "SVM"])
use_coral = st.sidebar.checkbox("Use CORAL Adaptation")
cross_subject = st.sidebar.checkbox("Cross-Subject Validation")

uploaded_file = None
if dataset == "Custom":
    uploaded_file = st.sidebar.file_uploader("Upload .npz or .mat EEG file", type=["npz", "mat"])

# Run pipeline button
if st.sidebar.button("Run Pipeline"):
    config = {"dataset": dataset, "model": model, "use_coral": use_coral}

    try:
        if dataset == "Custom":
            if uploaded_file is None:
                st.warning("⚠️ Please upload a file before running.")
                st.stop()
            X, y, groups = load_custom_file(uploaded_file)
        else:
            X, y, groups = load_dataset(dataset)

        pipeline = EEGPipeline(config=config, cross_subject=cross_subject)
        pipeline.X, pipeline.y, pipeline.groups = X, y, groups

        st.info("🔄 Loading and processing data...")
        results = pipeline.preprocess().extract_features().adapt().fit().evaluate()

        st.success("✅ Pipeline completed successfully!")
        st.metric("Mean", f"{results['mean_accuracy']:.3f}")
        st.metric("Std", f"{results['std_accuracy']:.3f}")
        st.line_chart(results["cv_scores"])

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"🚨 Unexpected error: {str(e)}")
