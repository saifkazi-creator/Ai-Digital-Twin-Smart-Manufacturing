# config.py — central configuration for all modules

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH        = "data/ai4i2020.csv"
MODEL_SAVE_PATH  = "models/best_model.pkl"
SCALER_SAVE_PATH = "models/scaler.pkl"
LOG_DIR          = "logs"
ALERT_LOG_PATH   = "logs/alerts.log"

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "mlruns/"
EXPERIMENT_NAME     = "SmartManufacturing_PredictiveMaintenance"

# ── Optuna ─────────────────────────────────────────────────────────────────────
OPTUNA_TRIALS_ML = 30   # RF and XGBoost
OPTUNA_TRIALS_DL = 20   # MLP

# ── Preprocessing ──────────────────────────────────────────────────────────────
RANDOM_STATE       = 42
TEST_SIZE          = 0.20
SMOTE_THRESHOLD    = 0.30   # apply SMOTE if minority class < 30%

# Feature columns (exact names as they appear in the CSV after stripping BOM)
FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COL = "Machine failure"
DROP_COLS  = ["UDI", "Product ID", "Type", "TWF", "HDF", "PWF", "OSF", "RNF"]

# ── Simulation ─────────────────────────────────────────────────────────────────
FAILURE_THRESHOLD    = 0.80   # probability above which status → Critical
OVERLOAD_THRESHOLD   = 0.40   # probability above which status → Overloaded
ALERT_COOLDOWN_STEPS = 10     # minimum steps between LLM alerts per machine
SIM_TIME_STEP        = 1      # wall-clock seconds between simulation ticks
HISTORY_WINDOW       = 100    # readings kept per machine for the dashboard

# ── LLM ────────────────────────────────────────────────────────────────────────
LLM_PROVIDER   = "gemini"                         # "gemini" | "openai"
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBEs3MyU-hTx4DVHySGqmyjPuIY4RpgccA")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
