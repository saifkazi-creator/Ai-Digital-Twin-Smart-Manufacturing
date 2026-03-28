# models/evaluate.py — evaluation utilities shared across all models

import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict:
    """Return a flat dict of all evaluation metrics."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
    return metrics


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str = "models/artifacts",
) -> str:
    """Plot and save confusion matrix as PNG; return file path."""
    os.makedirs(save_dir, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Failure"],
        yticklabels=["Normal", "Failure"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}")
    fig.tight_layout()
    path = os.path.join(save_dir, f"cm_{model_name}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str = "models/artifacts",
) -> str:
    """Write sklearn classification report to a .txt file; return path."""
    os.makedirs(save_dir, exist_ok=True)
    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Failure"], zero_division=0
    )
    path = os.path.join(save_dir, f"report_{model_name}.txt")
    with open(path, "w") as f:
        f.write(f"Classification Report — {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    return path


def print_metrics(metrics: dict, model_name: str) -> None:
    """Pretty-print evaluation metrics to stdout."""
    print(f"\n[evaluate] {model_name} results:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
