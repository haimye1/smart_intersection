
import torch
import joblib
import numpy as np
from autoencoder import WindowAE

def minmax(a):
    mn, mx=a.min(), a.max()
    return (a-mn)/(mx-mn+1e-12)

def infer(X, model_dir, input_dim, latent_dim=8):
    scaler=joblib.load(f"{model_dir}/scaler.pkl")
    iso=joblib.load(f"{model_dir}/iso_forest.pkl")
    Xs=scaler.transform(X)
    iso_score= -iso.decision_function(Xs)
    ae=WindowAE(input_dim,latent_dim)
    ae.load_state_dict(torch.load(f"{model_dir}/autoencoder.pt", map_location="cpu"))
    ae.eval()
    with torch.no_grad():
        Xr=ae(torch.tensor(Xs).float())
        ae_score=((Xr - torch.tensor(Xs).float())**2).mean(1).numpy()
    return minmax(iso_score), minmax(ae_score)
