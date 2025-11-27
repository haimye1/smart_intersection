
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_iso_forest(X_train, model_dir, random_state=42):
    iso = IsolationForest(n_estimators=200, contamination=0.01,
                          random_state=random_state)
    iso.fit(X_train)
    joblib.dump(iso, os.path.join(model_dir,"iso_forest.pkl"))
    return iso
