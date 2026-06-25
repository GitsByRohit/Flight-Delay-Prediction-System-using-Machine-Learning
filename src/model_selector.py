# ============================================
# Model Selector Script
# ============================================

import os
import joblib
import pandas as pd


# Paths
METRICS_CSV_PATH = "reports/model_metrics.csv"

MODEL_PATHS = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "SVM": "models/svm.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

BEST_MODEL_PATH = "models/best_model.pkl"


def main():
    print("Loading model metrics...")

    if not os.path.exists(METRICS_CSV_PATH):
        print("Metrics file not found. Run model_evaluation.py first.")
        return

    metrics_df = pd.read_csv(METRICS_CSV_PATH)

    if metrics_df.empty:
        print("Metrics file is empty.")
        return

    # Sort by F1-score descending
    metrics_df = metrics_df.sort_values(
        by="F1-Score",
        ascending=False
    )

    best_model_name = metrics_df.iloc[0]["Model"]

    print(f"Best model based on F1-Score: {best_model_name}")

    best_model_path = MODEL_PATHS.get(best_model_name)

    if not os.path.exists(best_model_path):
        print("Best model file not found.")
        return

    # ✅ Load model dictionary (not just model)
    model_data = joblib.load(best_model_path)

    model = model_data.get("model")
    scaler = model_data.get("scaler", None)
    threshold = model_data.get("threshold", 0.5)
    feature_columns = model_data.get("feature_columns")

    # Save deployment-ready model
    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "feature_columns": feature_columns
    }, BEST_MODEL_PATH)

    print(f"Best model saved at: {BEST_MODEL_PATH}")
    print("Model selection completed successfully.")


if __name__ == "__main__":
    main()
