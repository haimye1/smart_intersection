
from typing import Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

from .config import MODEL_DIR, RANDOM_STATE

def train_iso_forest(X_train: np.ndarray) -> IsolationForest:
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=RANDOM_STATE,
    )
    iso.fit(X_train)
    os.makedirs(MODEL_DIR, exist_ok=True)
    iso_path = os.path.join(MODEL_DIR, "isolation_forest_window.pkl")
    joblib.dump(iso, iso_path)
    return iso


def iso_scores(iso: IsolationForest, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_scores = -iso.decision_function(X_train)
    val_scores = -iso.decision_function(X_val)
    return train_scores, val_scores
