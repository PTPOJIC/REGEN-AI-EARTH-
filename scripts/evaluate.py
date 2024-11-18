import torch
from models.regenerative_model import RegenerativeModel
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load the trained model
model = RegenerativeModel(input_dim=6, output_dim=3)  # Adjust dimensions based on your data
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Load test data
test_features = pd.read_csv("data/generated_data/environmental_features.csv").values
test_targets = pd.read_csv("data/generated_data/nutrition_targets.csv").values

# Convert to tensors
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)

# Evaluate model on test data
with torch.no_grad():
    predictions = model(test_features_tensor)

# Convert predictions and targets to NumPy arrays for metrics calculation
predictions_np = predictions.numpy()
test_targets_np = test_targets_tensor.numpy()

# Calculate evaluation metrics
mse = mean_squared_error(test_targets_np, predictions_np)
r2 = r2_score(test_targets_np, predictions_np)

print(f"Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-Squared (R2): {r2:.4f}")

# Save predictions for analysis
predictions_df = pd.DataFrame(predictions_np, columns=["Predicted_Nitrogen", "Predicted_Phosphorus", "Predicted_Potassium"])
predictions_df.to_csv("data/generated_data/predictions.csv", index=False)
print("Predictions saved to data/generated_data/predictions.csv")
