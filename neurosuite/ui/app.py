# neurosuite/ui/app.py

import streamlit as st
from neurosuite import EEGPipeline

st.set_page_config(page_title="NeuroSuite GUI", layout="centered")

st.title("NeuroSuite EEG Modeling Toolkit")

# --- Sidebar Settings ---
st.sidebar.header("Configuration")

dataset = st.sidebar.selectbox("Select Dataset", ["DEAP", "SEED", "OpenNeuro"])
model = st.sidebar.selectbox("Machine Learning Model", ["svm", "rf", "xgb"])
use_coral = st.sidebar.checkbox("Enable CORAL Domain Adaptation", value=False)
cross_subject = st.sidebar.checkbox("Cross-Subject Validation", value=True)

if st.sidebar.button("Run Pipeline"):
    config = {"dataset": dataset, "model": model, "use_coral": use_coral}
    pipeline = EEGPipeline(config=config, cross_subject=cross_subject)

    try:
        # Try dataset loading only to validate existence
        _ = pipeline.load_data()

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



