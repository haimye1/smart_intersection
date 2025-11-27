
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from autoencoder import WindowAE

def train_autoencoder(X_train, X_val, model_dir, input_dim,
                      latent_dim=8, batch=128, epochs=20, lr=1e-3):
    device="cuda" if torch.cuda.is_available() else "cpu"
    ae=WindowAE(input_dim, latent_dim).to(device)
    ds_tr=TensorDataset(torch.tensor(X_train).float())
    ds_va=TensorDataset(torch.tensor(X_val).float())
    tr=DataLoader(ds_tr,batch,shuffle=True)
    va=DataLoader(ds_va,batch,shuffle=False)
    opt=torch.optim.Adam(ae.parameters(), lr=lr)
    crit=torch.nn.MSELoss()
    best=1e9
    for e in range(epochs):
        ae.train()
        tl=0
        for (xb,) in tr:
            xb=xb.to(device)
            opt.zero_grad()
            rec=ae(xb)
            loss=crit(rec,xb)
            loss.backward()
            opt.step()
            tl+=loss.item()*xb.size(0)
        tl/=len(ds_tr)
        ae.eval()
        vl=0
        with torch.no_grad():
            for (xb,) in va:
                xb=xb.to(device)
                rec=ae(xb)
                vl+=crit(rec,xb).item()*xb.size(0)
        vl/=len(ds_va)
        if vl<best:
            best=vl
            torch.save(ae.state_dict(), os.path.join(model_dir,"autoencoder.pt"))
    return True
