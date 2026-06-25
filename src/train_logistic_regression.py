# ============================================
# Logistic Regression with Scaling + Threshold
# ============================================

import os
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from config import (
    FINAL_DATA_PATH,
    LOGISTIC_MODEL_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN
)

DECISION_THRESHOLD = 0.4


def load_final_data():
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


def scale_features(X_train, X_test):
    print("\nApplying Feature Scaling...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE,
        solver="lbfgs"
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

    print("\n📊 Logistic Regression Performance:")
    print(f"Decision Threshold: {DECISION_THRESHOLD}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


def save_model(model, scaler, feature_columns):
    os.makedirs(os.path.dirname(LOGISTIC_MODEL_PATH), exist_ok=True)

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "threshold": DECISION_THRESHOLD,
        "feature_columns": feature_columns
    }, LOGISTIC_MODEL_PATH)

    print(f"\nModel saved at: {LOGISTIC_MODEL_PATH}")


def main():
    df = load_final_data()

    X, (X_train, X_test, y_train, y_test) = split_data(df)

    X_train_balanced, y_train_balanced = apply_random_undersampling(X_train, y_train)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_balanced, X_test)

    model = train_model(X_train_scaled, y_train_balanced)

    evaluate_model(model, X_test_scaled, y_test)

    save_model(model, scaler, X.columns.tolist())

    print("\nLogistic Regression training completed successfully!")


if __name__ == "__main__":
    main()
