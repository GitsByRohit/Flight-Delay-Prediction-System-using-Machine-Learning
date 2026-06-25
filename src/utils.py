# ============================================
# Utility Functions for ML Pipeline
# ============================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def apply_random_undersampling(X_train, y_train, target_column, random_state):
    """
    Perform random undersampling on training data only.
    """

    train_df = pd.concat([X_train, y_train], axis=1)

    df_majority = train_df[train_df[target_column] == 1]
    df_minority = train_df[train_df[target_column] == 0]

    df_majority_sampled = df_majority.sample(
        n=len(df_minority),
        random_state=random_state
    )

    df_balanced = pd.concat([df_majority_sampled, df_minority])

    X_balanced = df_balanced.drop(columns=[target_column])
    y_balanced = df_balanced[target_column]

    return X_balanced, y_balanced


def scale_features(X_train, X_test):
    """
    Apply standard scaling.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def predict_with_threshold(model, X, threshold):
    """
    Apply probability or decision threshold.
    """

    # Probability-based models
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)

    # SVM decision function
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores >= threshold).astype(int)

    # Fallback
    else:
        return model.predict(X)


def calculate_metrics(y_true, y_pred):
    """
    Compute evaluation metrics.
    """

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm
