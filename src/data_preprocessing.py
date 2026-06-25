import pandas as pd
from config import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    DELAY_THRESHOLD,
    TARGET_COLUMN
)

print(">>> RUNNING UPDATED PREPROCESSING FILE <<<")


def preprocess_data():
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Dropping missing required columns...")
    df = df.dropna(subset=["ArrDelay", "DepTime", "Date"])

    print("Creating time features...")

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["Date"])

    df["Month"] = df["Date"].dt.month
    df["DayOfMonth"] = df["Date"].dt.day
    df["Hour"] = (df["DepTime"] // 100).astype(int)

    print("Creating target column...")
    df[TARGET_COLUMN] = (df["ArrDelay"] > DELAY_THRESHOLD).astype(int)

    print("Saving cleaned dataset...")
    df.to_csv(CLEANED_DATA_PATH, index=False)

    print("Columns in cleaned dataset:")
    print(df.columns)


if __name__ == "__main__":
    preprocess_data()
