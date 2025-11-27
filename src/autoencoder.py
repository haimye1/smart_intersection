
import torch
import torch.nn as nn

class WindowAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,latent_dim)
        )
        self.decoder=nn.Sequential(
            nn.Linear(latent_dim,32), nn.ReLU(),
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,input_dim)
        )
    def forward(self,x):
        z=self.encoder(x)
        return self.decoder(z)
