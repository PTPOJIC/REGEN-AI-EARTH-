import torch
from torch.utils.data import DataLoader, TensorDataset
from models.regenerative_model import RegenerativeModel
from models.model_utils import train_model, save_model
import pandas as pd

# Load data
features = pd.read_csv("data/generated_data/environmental_features.csv").values
targets = pd.read_csv("data/generated_data/nutrition_targets.csv").values

# Convert to tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

# Prepare dataset and dataloader
dataset = TensorDataset(features_tensor, targets_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = features.shape[1]
output_dim = targets.shape[1]
model = RegenerativeModel(input_dim, output_dim)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, dataloader, criterion, optimizer, epochs=20)

# Save the model
save_model(model)
