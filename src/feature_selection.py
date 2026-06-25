# ============================================
# Feature Selection & Encoding Script
# ============================================

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config import (
    CLEANED_DATA_PATH,
    FINAL_DATA_PATH,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN
)


def load_cleaned_data():
    print("Loading cleaned dataset...")
    return pd.read_csv(CLEANED_DATA_PATH)


def select_required_columns(df):
    """
    Keep only finalized ML features + target
    """
    required_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN]
    return df[required_columns]


def encode_categorical_features(df):
    """
    One-hot encode categorical features
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    encoded_data = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    )

    return encoded_df


def feature_engineering_pipeline():
    df = load_cleaned_data()

    print("Selecting required columns...")
    df = select_required_columns(df)

    print("Encoding categorical features...")
    encoded_df = encode_categorical_features(df)

    print("Combining numerical + encoded features...")

    final_df = pd.concat(
        [
            encoded_df.reset_index(drop=True),
            df[NUMERICAL_FEATURES].reset_index(drop=True),
            df[[TARGET_COLUMN]].reset_index(drop=True)
        ],
        axis=1
    )

    print("Saving final dataset...")
    final_df.to_csv(FINAL_DATA_PATH, index=False)

    print("✅ Feature engineering completed successfully!")
    print(f"Saved at: {FINAL_DATA_PATH}")


if __name__ == "__main__":
    feature_engineering_pipeline()
