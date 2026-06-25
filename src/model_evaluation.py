# ============================================
# Model Evaluation & Comparison Script
# ============================================

import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

from config import (
    FINAL_DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN
)

# Model paths
MODEL_PATHS = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "SVM": "models/svm.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

# Report paths (UPDATED — stored in reports/)
METRICS_CSV_PATH = "reports/model_metrics.csv"
REPORT_PATH = "reports/comparison_report.txt"


def load_test_data():
    df = pd.read_csv(FINAL_DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_test, y_test


def evaluate_model(model_obj, X_test, y_test):
    """
    Handles:
    - Models saved as dict (with scaler/threshold)
    - Plain sklearn models
    """

    if isinstance(model_obj, dict):
        model = model_obj.get("model")
        scaler = model_obj.get("scaler")
        threshold = model_obj.get("threshold", 0.5)
    else:
        model = model_obj
        scaler = None
        threshold = 0.5

    # Apply scaling if available
    if scaler is not None:
        X_test = scaler.transform(X_test)

    # Use predict_proba if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        y_pred = (probabilities >= threshold).astype(int)

    # Use decision_function if available (e.g., SVM)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_pred = (scores >= threshold).astype(int)

    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, prec, rec, f1, cm


def main():
    print("Loading test data...")
    X_test, y_test = load_test_data()

    results = []
    report_lines = []

    report_lines.append("MODEL COMPARISON REPORT")
    report_lines.append("=" * 60)

    for model_name, path in MODEL_PATHS.items():

        if not os.path.exists(path):
            print(f"{model_name} model not found. Skipping...")
            continue

        print(f"Evaluating {model_name}...")

        model_obj = joblib.load(path)

        acc, prec, rec, f1, cm = evaluate_model(
            model_obj, X_test, y_test
        )

        results.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4)
        })

        report_lines.append(f"\n{model_name}")
        report_lines.append("-" * 40)
        report_lines.append(f"Accuracy   : {acc:.4f}")
        report_lines.append(f"Precision  : {prec:.4f}")
        report_lines.append(f"Recall     : {rec:.4f}")
        report_lines.append(f"F1-Score   : {f1:.4f}")
        report_lines.append(f"Confusion Matrix:\n{cm}")

    # Create DataFrame
    metrics_df = pd.DataFrame(results)

    # Sort by F1-score
    metrics_df = metrics_df.sort_values(
        by="F1-Score",
        ascending=False
    )

    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)

    # Save CSV
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    # Determine best model
    best_model = metrics_df.iloc[0]["Model"]

    report_lines.append("\n" + "=" * 60)
    report_lines.append(f"\nBest Model Based on F1-Score: {best_model}")

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print("\nModel evaluation completed successfully.")
    print(f"Metrics saved at: {METRICS_CSV_PATH}")
    print(f"Report saved at: {REPORT_PATH}")
    print(f"Best Model: {best_model}")


if __name__ == "__main__":
    main()
