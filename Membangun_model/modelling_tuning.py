import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set tracking lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Zara Sales Tuning")

# Load data
df = pd.read_csv("Membangun_model/dataset_preprocessing/zara_ready.csv")
X = df.drop(columns="Sales Volume")
y = df["Sales Volume"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10]
}

# Loop manual tuning
for n in grid["n_estimators"]:
    for d in grid["max_depth"]:
        with mlflow.start_run():
            model = RandomForestRegressor(n_estimators=n, max_depth=d)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            # Manual logging
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", d)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(model, "model")

            print(f"Logged: n_estimators={n}, max_depth={d}, mse={mse:.2f}, r2={r2:.2f}, mae={mae:.2f}")
