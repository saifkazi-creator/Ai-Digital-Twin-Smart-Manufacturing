# models/tune.py — Optuna hyperparameter tuning for RF, XGBoost, and MLP

import os
import sys
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import OPTUNA_TRIALS_ML, OPTUNA_TRIALS_DL, RANDOM_STATE


# ── Random Forest ──────────────────────────────────────────────────────────────

def tune_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> dict:
    """Tune Random Forest with Optuna; optimise macro F1 via 3-fold CV."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
        }
        model = RandomForestClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1
        )
        return scores.mean()

    print("\n[tune] Starting Random Forest Optuna study...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS_ML, show_progress_bar=True)

    print(f"\n[tune] RF best trial:")
    print(f"  F1 (macro): {study.best_value:.4f}")
    print(f"  Params:     {study.best_params}")
    return study.best_params


# ── XGBoost ────────────────────────────────────────────────────────────────────

def tune_xgboost(
    X_train: np.ndarray, y_train: np.ndarray
) -> dict:
    """Tune XGBoost with Optuna; optimise macro F1 via 3-fold CV."""
    # Compute scale_pos_weight for imbalanced data
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight":  spw,
            "use_label_encoder": False,
            "eval_metric":       "logloss",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
        }
        model = XGBClassifier(**params, verbosity=0)
        scores = cross_val_score(
            model, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1
        )
        return scores.mean()

    print("\n[tune] Starting XGBoost Optuna study...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS_ML, show_progress_bar=True)

    print(f"\n[tune] XGB best trial:")
    print(f"  F1 (macro): {study.best_value:.4f}")
    print(f"  Params:     {study.best_params}")
    return study.best_params


# ── MLP (PyTorch) ──────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def _train_mlp_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _eval_mlp(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        probs = model(X_t).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return f1_score(y, preds, average="macro", zero_division=0)


def tune_mlp(
    X_train: np.ndarray, y_train: np.ndarray
) -> dict:
    """Tune PyTorch MLP with Optuna; optimise validation macro F1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    # Simple 80/20 inner split for validation during tuning
    split = int(0.8 * len(X_train))
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    def objective(trial: optuna.Trial) -> float:
        h1          = trial.suggest_int("h1", 64, 256)
        h2          = trial.suggest_int("h2", 32, 128)
        dropout     = trial.suggest_float("dropout", 0.1, 0.5)
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

        model     = MLPClassifier(input_dim, h1, h2, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X_tr, dtype=torch.float32)
        y_t = torch.tensor(y_tr, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        best_val_f1  = 0.0
        patience_cnt = 0
        patience     = 5

        for epoch in range(50):
            _train_mlp_epoch(model, loader, criterion, optimizer, device)
            val_f1 = _eval_mlp(model, X_val, y_val, device)

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break   # early stopping

        return best_val_f1

    print("\n[tune] Starting MLP Optuna study...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS_DL, show_progress_bar=True)

    print(f"\n[tune] MLP best trial:")
    print(f"  Val F1 (macro): {study.best_value:.4f}")
    print(f"  Params:         {study.best_params}")
    return study.best_params


# ── Entry point ────────────────────────────────────────────────────────────────

def run_tuning(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[dict, dict, dict]:
    """Run all three Optuna studies and return best params for each model."""
    rf_params  = tune_random_forest(X_train, y_train)
    xgb_params = tune_xgboost(X_train, y_train)
    mlp_params = tune_mlp(X_train, y_train)
    print("\n[tune] All studies complete.")
    return rf_params, xgb_params, mlp_params
