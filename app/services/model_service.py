import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/best_model.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data.get("model")
scaler = model_data.get("scaler")
threshold = model_data.get("threshold", 0.5)
feature_columns = model_data.get("feature_columns")


def predict_delay(input_df: pd.DataFrame):

    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    if scaler is not None:
        input_df = scaler.transform(input_df)

    # ==========================
    # SVM CASE → USE RAW SCORE
    # ==========================
    if hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):

        score = model.decision_function(input_df)[0]

        prediction = 1 if score >= threshold else 0

        prob_delay = 1 / (1 + np.exp(-score))  # only for display

    # ==========================
    # PROBABILITY MODELS
    # ==========================
    else:

        prob_delay = model.predict_proba(input_df)[:, 1][0]

        prediction = 1 if prob_delay >= threshold else 0

    # ==========================
    # CONFIDENCE
    # ==========================
    confidence = prob_delay if prediction == 1 else (1 - prob_delay)

    confidence = round(confidence * 100, 2)

    return prediction, confidence