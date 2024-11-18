Precognition AI for Planting
============================

A machine learning project using Regenerative AI to analyze earth voltage and nutrition data, 
predict optimal planting conditions, and guide sustainable agricultural practices.

Project Overview
----------------
This project leverages synthetic data generation and a custom-trained AI model to simulate and predict the relationship 
between environmental factors (e.g., soil moisture, pH, temperature) and soil nutrients (nitrogen, phosphorus, potassium).  
The ultimate goal is to provide insights for optimal planting conditions using predictive modeling and data analysis.

Features
--------
1. **Synthetic Dataset Generator**:  
   Generates environmental and soil nutrition data, simulating real-world planting conditions.

2. **Custom AI Model**:  
   A neural network trained to predict soil nutrients from environmental features.

3. **Evaluation and Visualization**:  
   Includes scripts to evaluate model performance and visualize insights for decision-making.

4. **Planting Precognition**:  
   Makes predictions for unseen environmental data to guide planting decisions.

Directory Structure
-------------------
::

    precognition_ai_planting/
    │
    ├── data/
    │   ├── generated_data/
    │   └── dataset_generator.py         # Generates synthetic dataset
    │
    ├── models/
    │   ├── regenerative_model.py        # Neural network definition
    │   └── model_utils.py               # Training utilities
    │
    ├── scripts/
    │   ├── train.py                     # Train the AI model
    │   ├── evaluate.py                  # Evaluate model performance
    │   ├── predict.py                   # Make predictions for new data
    │   └── visualization.py             # Visualize relationships in data
    │
    ├── requirements.txt                 # Dependencies
    ├── README.md                        # Project description
    └── main.py                          # Entry point for dataset generation, training, and prediction

Setup and Installation
----------------------
### Prerequisites

- Python 3.8+
- Libraries: ``torch``, ``numpy``, ``pandas``, ``matplotlib``, ``seaborn``

### Install Dependencies

.. code-block:: bash

    pip install -r requirements.txt

Usage
-----
1. Generate Dataset
~~~~~~~~~~~~~~~~~~~

Run the dataset generator to create synthetic data for training and testing:

.. code-block:: bash
    mkdir data/generated_data
    chmod -R 755 data/generated_data
    python data/dataset_generator.py

This generates two CSV files:

- **`environmental_features.csv`**: Simulated environmental data (e.g., soil moisture, temperature).
- **`nutrition_targets.csv`**: Corresponding nutrient levels (nitrogen, phosphorus, potassium).

2. Train the Model
~~~~~~~~~~~~~~~~~~
Train the neural network using the generated dataset:

.. code-block:: bash

    python scripts/train.py

After training, the model is saved as ``trained_model.pth``.

3. Evaluate the Model
~~~~~~~~~~~~~~~~~~~~~
Evaluate the model's performance using test data:

.. code-block:: bash

    python scripts/evaluate.py

The script outputs evaluation metrics like:

- **Mean Squared Error (MSE)**
- **R-Squared (R2)**

Predictions are saved to ``data/generated_data/predictions.csv``.

4. Make Predictions
~~~~~~~~~~~~~~~~~~~
Predict soil nutrient levels for new environmental conditions:

.. code-block:: bash

    python scripts/predict.py

Edit the ``new_data`` in the script to input custom environmental features.

5. Visualize Data
~~~~~~~~~~~~~~~~~
Visualize relationships between environmental features and soil nutrients:

.. code-block:: bash

    python scripts/visualization.py

Example Use Case
----------------
1. Simulate environmental data for a field using sensors or manually.
2. Predict soil nutrients using the trained AI model.
3. Optimize planting strategies based on predicted nutrient availability.

Project Goals
-------------
- Provide actionable insights for precision agriculture.
- Use AI to promote sustainable farming by predicting optimal planting conditions.

Future Enhancements
-------------------
- Incorporate real-world datasets for more accurate predictions.
- Extend to support real-time sensor integration.
- Add recommendations for specific crops based on soil and environmental conditions.

License
-------
This project is licensed under the MIT License.

Contributors
------------
- **Kreatif Indonesia**  
  [PT Pusakaofjava Indonesia Corp 

Feel free to contribute to this project by submitting issues or pull requests!
