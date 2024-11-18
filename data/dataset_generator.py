import numpy as np
import pandas as pd
import os

def generate_earth_data(num_samples=100000):
    """
    Generate synthetic Earth Voltage and Nutrition dataset for planting.
    """
    # Simulating Earth Voltage parameters
    soil_moisture = np.random.uniform(0.1, 1.0, num_samples)  # Moisture: 10%-100%
    soil_ph = np.random.uniform(5.0, 8.5, num_samples)        # pH for most plants
    soil_conductivity = np.random.uniform(0.5, 2.5, num_samples)  # mS/cm conductivity

    # Environmental factors
    temperature = np.random.uniform(15, 35, num_samples)  # Celsius
    sunlight_hours = np.random.uniform(6, 12, num_samples)  # Hours/day
    rainfall = np.random.uniform(0, 50, num_samples)  # mm/day

    # Nutrition levels (target variables)
    nitrogen = soil_moisture * np.random.uniform(10, 30) + np.random.normal(0, 2, num_samples)
    phosphorus = soil_ph * np.random.uniform(5, 20) + np.random.normal(0, 1, num_samples)
    potassium = soil_conductivity * np.random.uniform(15, 40) + np.random.normal(0, 3, num_samples)

    # Combine features and targets
    features = pd.DataFrame({
        "soil_moisture": soil_moisture,
        "soil_ph": soil_ph,
        "soil_conductivity": soil_conductivity,
        "temperature": temperature,
        "sunlight_hours": sunlight_hours,
        "rainfall": rainfall
    })
    targets = pd.DataFrame({
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium
    })

    return features, targets

if __name__ == "__main__":
    # Generate and save the dataset
    features, targets = generate_earth_data()
    features.to_csv("data/generated_data/environmental_features.csv", index=False)
    targets.to_csv("data/generated_data/nutrition_targets.csv", index=False)
    print("Dataset generated and saved.")
# Ensure the target directory exists
os.makedirs("data/generated_data", exist_ok=True)
