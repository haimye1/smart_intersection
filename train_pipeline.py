
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    DATA_PATH,
    MODEL_DIR,
    WINDOW_SIZE,
    WINDOW_STEP,
    BATCH_SIZE,
    AE_LATENT_DIM,
    AE_EPOCHS,
    AE_LR,
    ensure_dirs,
)
from .data_loader import load_dataset
from .windowing import build_windows, aggregate_window_features
from .features import scale_features, train_val_split_time
from .iso_forest import train_iso_forest, iso_scores
from .ae_model import WindowAutoencoder

sns.set(style="darkgrid")


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
) -> Tuple[str, np.ndarray, np.ndarray]:
    device = _device()
    ae = WindowAutoencoder(input_dim=input_dim, latent_dim=AE_LATENT_DIM).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=AE_LR)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    best_path = os.path.join(MODEL_DIR, "autoencoder_best.pt")

    for epoch in range(1, AE_EPOCHS + 1):
        ae.train()
        train_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_rec = ae(xb)
            loss = criterion(x_rec, xb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_rec = ae(xb)
                loss = criterion(x_rec, xb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[AE] Epoch {epoch:02d}/{AE_EPOCHS} | train={train_loss:.6f} | val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ae.state_dict(), best_path)

    ae.load_state_dict(torch.load(best_path, map_location=device))
    ae.eval()
    with torch.no_grad():
        X_all = np.vstack([X_train, X_val])
        X_t = torch.tensor(X_all, dtype=torch.float32).to(device)
        X_rec = ae(X_t)
        mse = ((X_rec - X_t) ** 2).mean(dim=1).cpu().numpy()

    return best_path, mse[: len(X_train)], mse[len(X_train) :]


def _plot_heatmap(anomaly_df: pd.DataFrame, out_dir: str):
    heat = np.stack(
        [
            anomaly_df["iso_score"].values,
            anomaly_df["ae_score"].values,
            anomaly_df["combined"].values,
        ],
        axis=0,
    )
    plt.figure(figsize=(14, 3))
    sns.heatmap(
        heat,
        cmap="magma",
        cbar=True,
        yticklabels=["IsolationForest", "Autoencoder", "Combined"],
    )
    plt.xlabel("Window index (time →)")
    plt.title("Smart Intersection – anomaly scores over the day")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "anomaly_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved heatmap to {out_path}")


def _plot_time_series(anomaly_df: pd.DataFrame, out_dir: str):
    time_center = pd.to_datetime(anomaly_df["center_time"])
    plt.figure(figsize=(14, 4))
    plt.plot(time_center, anomaly_df["iso_score"], label="IF anomaly", alpha=0.6)
    plt.plot(time_center, anomaly_df["ae_score"], label="AE anomaly", alpha=0.6)
    plt.plot(time_center, anomaly_df["combined"], label="Combined", linewidth=2)
    plt.xticks(rotation=45)
    plt.ylabel("Normalized anomaly score")
    plt.title("Smart Intersection – anomaly score over time")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "anomaly_time_series.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved time-series plot to {out_path}")


def run_full_pipeline() -> pd.DataFrame:
    ensure_dirs()
    print(f"[PIPELINE] Loading dataset from: {DATA_PATH}")
    df = load_dataset(DATA_PATH)

    feature_cols = [
        "phase_id",
        "time_in_phase",
        "queue_NS",
        "queue_EW",
        "avg_speed_NS",
        "avg_speed_EW",
        "ped_wait_NS",
        "ped_wait_EW",
        "v2x_msg_rate",
    ]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    X_raw = df[feature_cols].astype(float).values
    t_sec = df["time_sec"].values
    t_dt = df["time_of_day"].values

    print("[PIPELINE] Building windows...")
    windows, meta_df = build_windows(
        X_raw,
        t_sec,
        t_dt,
        window_size=WINDOW_SIZE,
        step=WINDOW_STEP,
    )
    print("[PIPELINE] windows shape:", windows.shape)

    X_win = aggregate_window_features(windows)
    print("[PIPELINE] window features shape:", X_win.shape)

    print("[PIPELINE] Scaling features...")
    X_scaled, _ = scale_features(X_win)

    print("[PIPELINE] Splitting train/val...")
    X_train, X_val, train_size = train_val_split_time(X_scaled, frac=0.8)

    print("[PIPELINE] Training IsolationForest...")
    iso = train_iso_forest(X_train)
    iso_train, iso_val = iso_scores(iso, X_train, X_val)

    print("[PIPELINE] Training Autoencoder...")
    input_dim = X_scaled.shape[1]
    _, ae_train, ae_val = _train_autoencoder(X_train, X_val, input_dim)

    iso_all = np.concatenate([iso_train, iso_val])
    ae_all = np.concatenate([ae_train, ae_val])

    iso_norm = _minmax(iso_all)
    ae_norm = _minmax(ae_all)
    combined = 0.5 * (iso_norm + ae_norm)

    anomaly_df = meta_df.copy()
    anomaly_df["iso_score"] = iso_norm
    anomaly_df["ae_score"] = ae_norm
    anomaly_df["combined"] = combined

    os.makedirs(MODEL_DIR, exist_ok=True)
    csv_path = os.path.join(MODEL_DIR, "anomaly_scores.csv")
    anomaly_df.to_csv(csv_path, index=False)
    print(f"[PIPELINE] Saved anomaly scores to {csv_path}")

    _plot_heatmap(anomaly_df, MODEL_DIR)
    _plot_time_series(anomaly_df, MODEL_DIR)

    print("[PIPELINE] Done.")
    return anomaly_df
