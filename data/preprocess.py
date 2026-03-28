# data/preprocess.py — data loading, cleaning, balancing, and scaling

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Allow imports from the project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_PATH, SCALER_SAVE_PATH, FEATURE_COLS, TARGET_COL, DROP_COLS,
    RANDOM_STATE, TEST_SIZE, SMOTE_THRESHOLD,
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV; strip BOM and whitespace from column names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are not features or target."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report and fill any missing values with column median."""
    missing = df.isnull().sum()
    if missing.any():
        print(f"[preprocess] Missing values detected:\n{missing[missing > 0]}")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("[preprocess] No missing values found.")
    return df


def check_class_balance(y: pd.Series) -> float:
    """Return minority class ratio and print distribution."""
    counts = y.value_counts()
    minority_ratio = counts.min() / counts.sum()
    print(f"[preprocess] Class distribution:\n{counts.to_dict()}")
    print(f"[preprocess] Minority class ratio: {minority_ratio:.3f}")
    return minority_ratio


def apply_smote(
    X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority class using SMOTE."""
    print("[preprocess] Applying SMOTE to balance classes...")
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    unique, counts = np.unique(y_res, return_counts=True)
    print(f"[preprocess] Post-SMOTE distribution: {dict(zip(unique, counts))}")
    return X_res, y_res


def run_preprocessing() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Scaled feature arrays and label arrays ready for model training.
    """
    # 1. Load
    df = load_data()
    print(f"[preprocess] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} cols")

    # 2. Drop irrelevant columns
    df = drop_irrelevant_columns(df)

    # 3. Handle missing values
    df = handle_missing_values(df)

    # 4. Separate features and target
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 5. Train/test split (stratified, before SMOTE to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[preprocess] Split → train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    # 6. Class balance check + SMOTE on training set only
    minority_ratio = check_class_balance(pd.Series(y_train))
    if minority_ratio < SMOTE_THRESHOLD:
        X_train, y_train = apply_smote(X_train, y_train)

    # 7. Scale features — fit ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("[preprocess] Features scaled with StandardScaler.")

    # 8. Persist scaler for use during simulation inference
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"[preprocess] Scaler saved → {SCALER_SAVE_PATH}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_preprocessing()
    print(f"\n[preprocess] Final shapes:")
    print(f"  X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"  X_test={X_test.shape},  y_test={y_test.shape}")
