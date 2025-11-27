
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

from .config import MODEL_DIR

def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    os.makedirs(MODEL_DIR, exist_ok=True)
    scaler_path = os.path.join(MODEL_DIR, "scaler_window_features.pkl")
    joblib.dump(scaler, scaler_path)
    return X_scaled, scaler


def train_val_split_time(X_scaled: np.ndarray, frac: float = 0.8):
    num_windows = X_scaled.shape[0]
    train_size = int(num_windows * frac)
    X_train = X_scaled[:train_size]
    X_val = X_scaled[train_size:]
    return X_train, X_val, train_size
