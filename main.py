# main.py — orchestrates the full smart manufacturing pipeline

import os
import sys
import subprocess

# Ensure project root is on the path
sys.path.append(os.path.dirname(__file__))


def main():
    print("=" * 60)
    print(" Smart Manufacturing — AI Digital Twin Pipeline")
    print("=" * 60)

    # ── Step 1: Preprocessing ──────────────────────────────────────────────────
    print("\n[Step 1/4] Data preprocessing...")
    from data.preprocess import run_preprocessing
    X_train, X_test, y_train, y_test = run_preprocessing()

    # ── Step 2: Hyperparameter tuning ─────────────────────────────────────────
    print("\n[Step 2/4] Optuna hyperparameter tuning (this may take ~30 min)...")
    from models.tune import run_tuning
    rf_params, xgb_params, mlp_params = run_tuning(X_train, y_train)

    # ── Step 3: Training + MLflow tracking ────────────────────────────────────
    print("\n[Step 3/4] Training all models with MLflow tracking...")
    from models.train import run_training
    best_model = run_training(
        X_train, X_test, y_train, y_test,
        rf_params, xgb_params, mlp_params,
    )

    print("\n[Step 3/4] Training complete.")
    print("  View experiments: mlflow ui  →  http://localhost:5000")

    # ── Step 4: Launch Streamlit dashboard ────────────────────────────────────
    print("\n[Step 4/4] Launching Streamlit dashboard...")
    print("  Dashboard URL: http://localhost:8501")
    print("  Press Ctrl+C to stop.\n")

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard_path,
         "--server.port", "8501",
         "--server.headless", "true"],
        check=True,
    )


if __name__ == "__main__":
    main()
