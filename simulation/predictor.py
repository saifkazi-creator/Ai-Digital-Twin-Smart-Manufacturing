# simulation/predictor.py — load best model + scaler and run real-time inference

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURE_COLS


class RealTimePredictor:
    """
    Wraps the trained best model and StandardScaler for single-row inference.

    Usage
    -----
    predictor = RealTimePredictor()
    prob = predictor.predict(snapshot_dict)   # returns float in [0, 1]
    """

    def __init__(
        self,
        model_path:  str = MODEL_SAVE_PATH,
        scaler_path: str = SCALER_SAVE_PATH,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[predictor] Model not found at '{model_path}'. "
                "Run main.py first to train and save the model."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"[predictor] Scaler not found at '{scaler_path}'. "
                "Run main.py first to preprocess the data."
            )

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Feature order must match training
        self._feature_keys = [
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
        ]
        print(f"[predictor] Model  loaded from: {model_path}")
        print(f"[predictor] Scaler loaded from: {scaler_path}")

    def predict(self, snapshot: dict) -> float:
        """
        Predict failure probability from a machine snapshot dict.

        Parameters
        ----------
        snapshot : dict
            Must contain keys matching self._feature_keys.

        Returns
        -------
        float
            Probability of machine failure in [0, 1].
        """
        try:
            row = np.array(
                [[snapshot[k] for k in self._feature_keys]], dtype=np.float32
            )
            row_scaled = self.scaler.transform(row)

            # Support both sklearn (predict_proba) and PyTorch (forward)
            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba(row_scaled)[0][1])
            else:
                # PyTorch MLPClassifier
                import torch
                self.model.eval()
                with torch.no_grad():
                    t    = torch.tensor(row_scaled, dtype=torch.float32)
                    prob = float(self.model(t).item())

            return min(max(prob, 0.0), 1.0)   # clamp to [0, 1]

        except Exception as e:
            print(f"[predictor] Inference error: {e}")
            return 0.0


# ── Module-level singleton factory ─────────────────────────────────────────────

_predictor_instance: RealTimePredictor | None = None


def get_predictor() -> RealTimePredictor:
    """Return a cached singleton RealTimePredictor (loads model once)."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = RealTimePredictor()
    return _predictor_instance


def predict_failure_probability(snapshot: dict) -> float:
    """Convenience function usable as a callback in FactorySimulation."""
    return get_predictor().predict(snapshot)
