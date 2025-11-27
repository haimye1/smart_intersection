
import numpy as np
import pandas as pd

def build_windows(X, t_sec, t_dt, window_size=60, step=10):
    N, F = X.shape
    windows=[]
    meta=[]
    start=0
    while start+window_size<=N:
        end=start+window_size
        Xw=X[start:end]
        tw_sec=t_sec[start:end]
        tw_dt=t_dt[start:end]
        windows.append(Xw)
        meta.append({
            "start_idx":start,
            "end_idx":end,
            "center_sec":tw_sec[len(tw_sec)//2],
            "center_time":tw_dt[len(tw_dt)//2],
        })
        start+=step
    return np.stack(windows,0), pd.DataFrame(meta)

def aggregate_window_features(w):
    mean=w.mean(1)
    std=w.std(1)
    mn=w.min(1)
    mx=w.max(1)
    return np.concatenate([mean,std,mn,mx],1)
