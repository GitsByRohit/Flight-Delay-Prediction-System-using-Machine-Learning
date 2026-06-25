import pandas as pd
import joblib

MODEL_PATH = "models/best_model.pkl"

model_data = joblib.load(MODEL_PATH)
feature_columns = model_data["feature_columns"]


def preprocess_input(form_data: dict):

    input_dict = {
        "Airline": form_data["airline"],
        "Origin": form_data["origin_airport"].upper(),
        "Dest": form_data["destination_airport"].upper(),
        "Month": int(form_data["month"]),
        "DayOfMonth": int(form_data["day"]),
        "DayOfWeek": int(form_data["day_of_week"]),
        "Hour": int(form_data["departure_hour"]),
        "Distance": float(form_data["distance"])
    }

    df = pd.DataFrame([input_dict])

    # Apply one-hot encoding
    df = pd.get_dummies(df)

    # Identify missing columns required by the model
    missing_cols = [col for col in feature_columns if col not in df.columns]

    # Create all missing columns at once (fixes fragmentation warning)
    if missing_cols:
        missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, missing_df], axis=1)

    # Ensure correct column order
    df = df[feature_columns]

    return df