import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Synthetic data: [Temperature, CPU %]
X = [[60, 10], [85, 40], [105, 95], [110, 99], [70, 20], [95, 80]]
# Labels: [0 = Healthy, 1 = Crash]
y = [0, 0, 1, 1, 0, 1]

# Initialize and train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model locally (The "Artifact")
joblib.dump(model, "server_model.pkl")
print("Successfully trained! File 'server_model.pkl' created.")