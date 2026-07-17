"""
STEP 1: Register your existing model with MLflow.

What this does:
- Loads your already-trained .pkl model
- Creates a tracked "run" recording its parameters and accuracy metrics
- Saves a versioned, MLflow-managed copy of the model
- Registers it under a named model so you can always fetch "the current best one"

Run it with:
    python 1_log_model_to_mlflow.py

Then view results with:
    mlflow ui
    (open http://localhost:5000 in your browser)
"""

import os
import joblib
import mlflow
import mlflow.sklearn

# Points at a local folder by default; in Docker this is overridden
# to point at the mlflow-server container instead.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))

# ---- EDIT THESE THREE THINGS FOR YOUR REAL MODEL ----
MODEL_PATH = "E:\\Depi\\final project\\docker files (1)\\models\\rossmann_sales_model.pkl"          # path to your existing pickle file
MODEL_PARAMS = {                          # whatever hyperparameters you used
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [10, 14, 18, 22, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
MODEL_METRICS = {                         # your real evaluation results
    "rmse": 123.45,
    "mae": 668.378,
    "mape": 19.918,
    "mse":975543.148,
    'R2': 0.899
}
# ------------------------------------------------------

EXPERIMENT_NAME = "sales-forecasting"
REGISTERED_MODEL_NAME = "sales-forecast-model"

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="initial-model-registration") as run:
    # Load your existing model
    model = joblib.load(MODEL_PATH)

    # Log parameters (what settings produced this model)
    for key, value in MODEL_PARAMS.items():
        mlflow.log_param(key, value)

    # Log metrics (how good it is)
    for key, value in MODEL_METRICS.items():
        mlflow.log_metric(key, value)

    # Log and register the model itself
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Model registered as '{REGISTERED_MODEL_NAME}'")
    print("Run 'mlflow ui' and open http://localhost:5000 to view it.")
