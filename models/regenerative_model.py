import torch
import torch.nn as nn
import torch.nn as nn

class RegenerativeModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegenerativeModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
