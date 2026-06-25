# ===============================
# Project Configuration File
# ===============================

import os

# -------------------------------------------------
# Project Root Directory
# (Flight Delay Prediction/)
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -------------------------------------------------
# Data Paths
# -------------------------------------------------
RAW_DATA_PATH = os.path.join(
    ROOT_DIR, "data", "raw", "Flight_delay.csv"
)

CLEANED_DATA_PATH = os.path.join(
    ROOT_DIR, "data", "processed", "cleaned_data.csv"
)

FINAL_DATA_PATH = os.path.join(
    ROOT_DIR, "data", "processed", "final_data.csv"
)

# -------------------------------------------------
# Model Paths
# -------------------------------------------------
MODEL_DIR = os.path.join(ROOT_DIR, "models")

LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost.pkl")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# -------------------------------------------------
# Reports
# -------------------------------------------------
METRICS_PATH = os.path.join(
    ROOT_DIR, "reports", "model_metrics.csv"
)

# -------------------------------------------------
# ML Parameters
# -------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Delay threshold (minutes)
DELAY_THRESHOLD = 15

# -------------------------------------------------
# Feature Lists
# -------------------------------------------------
CATEGORICAL_FEATURES = [
    "Airline",
    "Origin",
    "Dest"
]

NUMERICAL_FEATURES = [
    "Month",
    "DayOfWeek",
    "DayOfMonth",
    "Hour",
    "Distance"
]

TARGET_COLUMN = "is_delayed"
