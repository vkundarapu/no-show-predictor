from pathlib import Path
import joblib
import pandas as pd

# Path to the saved model: project_root/models/no_show_model.joblib
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "no_show_model.joblib"

# Load once at import time
model = joblib.load(MODEL_PATH)

# These must match the features used in training
FEATURE_COLS = [
    "Age",
    "WaitingDays",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "Gender",
    "ApptWeekday",
]

def predict_no_show(features: dict) -> float:
    """
    features: dict with keys matching FEATURE_COLS
    returns: probability of no-show (float between 0 and 1)
    """
    # Ensure correct column order
    df = pd.DataFrame([features], columns=FEATURE_COLS)
    proba = model.predict_proba(df)[0, 1]
    return float(proba)
