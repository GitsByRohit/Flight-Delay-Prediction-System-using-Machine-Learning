# ============================================
# XGBoost with Undersampling + Threshold
# ============================================

import os
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier

from config import (
    FINAL_DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN
)

XGB_MODEL_PATH = os.path.join("models", "xgboost.pkl")

DECISION_THRESHOLD = 0.4


def load_data():
    print("Loading final dataset...")
    return pd.read_csv(FINAL_DATA_PATH)


def split_data(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print("\nOriginal Class Distribution:")
    print(y.value_counts())

    return X, train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def apply_random_undersampling(X_train, y_train):
    print("\nApplying Random Undersampling on training data...")

    train_df = pd.concat([X_train, y_train], axis=1)

    df_majority = train_df[train_df[TARGET_COLUMN] == 1]
    df_minority = train_df[train_df[TARGET_COLUMN] == 0]

    df_majority_sampled = df_majority.sample(
        n=len(df_minority),
        random_state=RANDOM_STATE
    )

    df_balanced = pd.concat([df_majority_sampled, df_minority])

    print("\nBalanced Training Distribution:")
    print(df_balanced[TARGET_COLUMN].value_counts())

    X_balanced = df_balanced.drop(columns=[TARGET_COLUMN])
    y_balanced = df_balanced[TARGET_COLUMN]

    return X_balanced, y_balanced


def train_model(X_train, y_train):
    print("Training XGBoost...")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model with custom threshold...")

    probabilities = model.predict_proba(X_test)[:, 1]
    y_pred = (probabilities >= DECISION_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 XGBoost Performance:")
    print(f"Decision Threshold: {DECISION_THRESHOLD}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


def save_model(model, feature_columns):
    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": model,
        "scaler": None,
        "threshold": DECISION_THRESHOLD,
        "feature_columns": feature_columns
    }, XGB_MODEL_PATH)

    print(f"\nModel saved at: {XGB_MODEL_PATH}")


def main():
    df = load_data()

    X, (X_train, X_test, y_train, y_test) = split_data(df)

    X_train_balanced, y_train_balanced = apply_random_undersampling(X_train, y_train)

    model = train_model(X_train_balanced, y_train_balanced)

    evaluate_model(model, X_test, y_test)

    save_model(model, X.columns.tolist())

    print("\nXGBoost training completed successfully!")


if __name__ == "__main__":
    main()
