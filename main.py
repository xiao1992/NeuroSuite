from neurosuite import EEGPipeline

def main():
    print("Starting NeuroSuite EEG Pipeline...")

    config = {
        "dataset": "DEAP",       # Options: "DEAP", "SEED", "OpenNeuro"
        "model": "xgb",          # Options: "svm", "rf", "xgb"
        "use_coral": True        # Apply CORAL domain adaptation
    }

    pipeline = EEGPipeline(config=config, cross_subject=True)
    results = pipeline.run_all()

    print("\n✅ Pipeline Finished")
    print(f"Mean Accuracy: {results['mean_accuracy']:.3f}")
    print(f"Std Accuracy: {results['std_accuracy']:.3f}")
    print(f"CV Scores: {results['cv_scores']}")

if __name__ == "__main__":
    main()
