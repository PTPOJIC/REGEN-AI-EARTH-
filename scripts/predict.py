import torch
from models.regenerative_model import RegenerativeModel
import pandas as pd
import sys
import os

# Print sys.path for debugging
print("Python Path:", sys.path)

# Add the root project directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the trained model
model = RegenerativeModel(input_dim=6, output_dim=3)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Example input data (new environmental data)
new_data = pd.DataFrame({
    "soil_moisture": [0.8],
    "soil_ph": [6.5],
    "soil_conductivity": [1.5],
    "temperature": [25],
    "sunlight_hours": [8],
    "rainfall": [20]
})
new_data_tensor = torch.tensor(new_data.values, dtype=torch.float32)

# Predict nutrient levels
with torch.no_grad():
    predictions = model(new_data_tensor)

print("Predicted Nutrient Levels:")
print(predictions)
