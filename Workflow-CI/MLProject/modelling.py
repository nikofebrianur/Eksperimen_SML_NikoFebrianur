import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=False, default="Workflow-CI/MLProject/dataset_preprocessing/zara_ready.csv")
args = parser.parse_args()

# Load data
print("Loading preprocessed dataset...")
df = pd.read_csv(args.data_path)
X = df.drop(columns="Sales Volume")
y = df["Sales Volume"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_example = X_train.iloc[0:5]

# Define and train model
n_estimators = 100
max_depth = 10

# Fit model
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# Manual logging
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)
mlflow.log_metric("mse", mse)
mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)

print(f"MSE: {mse}")
