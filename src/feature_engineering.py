
from sklearn.preprocessing import StandardScaler
import joblib
import os

def scale_features(X, model_dir):
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(model_dir,"scaler.pkl"))
    return Xs, scaler
