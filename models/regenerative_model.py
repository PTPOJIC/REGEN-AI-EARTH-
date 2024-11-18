import torch
import torch.nn as nn

class RegenerativeModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegenerativeModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output
      
