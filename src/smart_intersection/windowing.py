
from typing import Tuple
import numpy as np
import pandas as pd

def build_windows(
    X: np.ndarray,
    t_sec,
    t_dt,
    window_size: int = 60,
    step: int = 10,
) -> Tuple[np.ndarray, pd.DataFrame]:
    X = np.asarray(X)
    t_sec = np.asarray(t_sec)
    t_dt = np.asarray(t_dt)

    N, F = X.shape
    windows = []
    meta = []
    start = 0

    while start + window_size <= N:
        end = start + window_size
        Xw = X[start:end]
        tw_sec = t_sec[start:end]
        tw_dt = t_dt[start:end]

        windows.append(Xw)
        meta.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "start_sec": float(tw_sec[0]),
            "end_sec": float(tw_sec[-1]),
            "center_sec": float(tw_sec[len(tw_sec)//2]),
            "start_time": tw_dt[0],
            "end_time": tw_dt[-1],
            "center_time": tw_dt[len(tw_dt)//2],
        })
        start += step

    windows_arr = np.stack(windows, axis=0)
    meta_df = pd.DataFrame(meta)
    return windows_arr, meta_df


def aggregate_window_features(windows: np.ndarray) -> np.ndarray:
    w_mean = windows.mean(axis=1)
    w_std  = windows.std(axis=1)
    w_min  = windows.min(axis=1)
    w_max  = windows.max(axis=1)
    return np.concatenate([w_mean, w_std, w_min, w_max], axis=1)
