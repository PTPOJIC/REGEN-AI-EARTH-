import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
features = pd.read_csv("data/generated_data/environmental_features.csv")
targets = pd.read_csv("data/generated_data/nutrition_targets.csv")

# Combine features and targets
data = pd.concat([features, targets], axis=1)

# Pairplot to visualize relationships
sns.pairplot(data)
plt.show()
