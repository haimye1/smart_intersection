
from typing import Tuple
import os
import numpy as np
import joblib
import torch

from .config import MODEL_DIR, AE_LATENT_DIM
from .windowing import build_windows, aggregate_window_features
from .ae_model import WindowAutoencoder


def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def load_models(input_dim: int):
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_window_features.pkl"))
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest_window.pkl"))

    ae = WindowAutoencoder(input_dim=input_dim, latent_dim=AE_LATENT_DIM)
    ae.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder_best.pt"), map_location="cpu"))
    ae.eval()
    return scaler, iso, ae


def anomaly_for_batch(df_new, feature_cols, window_size: int, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_raw = df_new[feature_cols].astype(float).values
    t_sec = df_new["time_sec"].values
    t_dt = df_new["time_of_day"].values

    windows, meta = build_windows(X_raw, t_sec, t_dt, window_size=window_size, step=step)
    X_win = aggregate_window_features(windows)

    scaler, iso, ae = load_models(X_win.shape[1])
    X_scaled = scaler.transform(X_win)

    iso_scores = -iso.decision_function(X_scaled)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        X_rec = ae(X_t)
        ae_scores = ((X_rec - X_t) ** 2).mean(dim=1).numpy()

    iso_norm = _minmax(iso_scores)
    ae_norm = _minmax(ae_scores)
    combined = 0.5 * (iso_norm + ae_norm)
    return iso_norm, ae_norm, combined
