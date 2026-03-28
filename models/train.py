# models/train.py — MLflow-tracked training for RF, XGBoost, and MLP

import os
import sys
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_SAVE_PATH,
    SCALER_SAVE_PATH, RANDOM_STATE,
)
from models.tune import MLPClassifier, _train_mlp_epoch, _eval_mlp
from models.evaluate import (
    compute_metrics, save_confusion_matrix,
    save_classification_report, print_metrics,
)


# ── MLflow setup ───────────────────────────────────────────────────────────────

def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ── Random Forest ──────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    best_params: dict,
) -> tuple[RandomForestClassifier, dict]:
    _setup_mlflow()
    with mlflow.start_run(run_name="RF_Optuna_Tuned") as run:
        params = {**best_params, "random_state": RANDOM_STATE, "n_jobs": -1}
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        print_metrics(metrics, "Random Forest")

        # Artifacts
        cm_path     = save_confusion_matrix(y_test, y_pred, "RF")
        report_path = save_classification_report(y_test, y_pred, "RF")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "BestPredictiveMaintenance_Model")

        print(f"[train] RF run_id: {run.info.run_id}")
    return model, metrics


# ── XGBoost ────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    best_params: dict,
) -> tuple[XGBClassifier, dict]:
    _setup_mlflow()
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    with mlflow.start_run(run_name="XGB_Optuna_Tuned") as run:
        params = {
            **best_params,
            "scale_pos_weight": neg / pos if pos > 0 else 1.0,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        print_metrics(metrics, "XGBoost")

        cm_path     = save_confusion_matrix(y_test, y_pred, "XGB")
        report_path = save_classification_report(y_test, y_pred, "XGB")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "BestPredictiveMaintenance_Model")

        print(f"[train] XGB run_id: {run.info.run_id}")
    return model, metrics


# ── MLP ────────────────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    best_params: dict,
) -> tuple[MLPClassifier, dict]:
    _setup_mlflow()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    h1         = best_params["h1"]
    h2         = best_params["h2"]
    dropout    = best_params["dropout"]
    lr         = best_params["lr"]
    batch_size = best_params["batch_size"]

    with mlflow.start_run(run_name="MLP_Optuna_Tuned") as run:
        mlflow.log_params({**best_params, "epochs": 50, "early_stopping_patience": 5})

        model     = MLPClassifier(input_dim, h1, h2, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        # Split for early stopping validation
        val_split = int(0.9 * len(X_train))
        X_val, y_val = X_train[val_split:], y_train[val_split:]

        best_val_f1, patience_cnt = 0.0, 0
        best_state = None

        for epoch in range(50):
            loss = _train_mlp_epoch(model, loader, criterion, optimizer, device)
            val_f1 = _eval_mlp(model, X_val, y_val, device)

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 5:
                    print(f"[train] MLP early stop at epoch {epoch+1}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_te  = torch.tensor(X_test, dtype=torch.float32).to(device)
            probs = model(X_te).cpu().numpy()
        y_pred   = (probs >= 0.5).astype(int)
        metrics  = compute_metrics(y_test, y_pred, probs)

        mlflow.log_metrics(metrics)
        print_metrics(metrics, "MLP")

        cm_path     = save_confusion_matrix(y_test, y_pred, "MLP")
        report_path = save_classification_report(y_test, y_pred, "MLP")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)

        mlflow.pytorch.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "BestPredictiveMaintenance_Model")

        print(f"[train] MLP run_id: {run.info.run_id}")
    return model, metrics


# ── Model selection ────────────────────────────────────────────────────────────

def select_and_save_best_model(
    models_and_metrics: list[tuple],
) -> object:
    """
    Pick the model with highest F1 and save it as best_model.pkl.

    Parameters
    ----------
    models_and_metrics : list of (model_object, metrics_dict, name_str)
    """
    best = max(models_and_metrics, key=lambda x: x[1]["f1"])
    best_model, best_metrics, best_name = best

    print(f"\n[train] Best model: {best_name}  |  F1={best_metrics['f1']:.4f}")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"[train] Saved best model → {MODEL_SAVE_PATH}")
    return best_model


# ── Full training pipeline ─────────────────────────────────────────────────────

def run_training(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    rf_params:  dict,
    xgb_params: dict,
    mlp_params: dict,
) -> object:
    """Train all three models, log to MLflow, return the best model object."""
    print("\n[train] === Training Random Forest ===")
    rf_model,  rf_metrics  = train_random_forest(X_train, X_test, y_train, y_test, rf_params)

    print("\n[train] === Training XGBoost ===")
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test, xgb_params)

    print("\n[train] === Training MLP ===")
    mlp_model, mlp_metrics = train_mlp(X_train, X_test, y_train, y_test, mlp_params)

    best_model = select_and_save_best_model([
        (rf_model,  rf_metrics,  "Random Forest"),
        (xgb_model, xgb_metrics, "XGBoost"),
        (mlp_model, mlp_metrics, "MLP"),
    ])
    return best_model
