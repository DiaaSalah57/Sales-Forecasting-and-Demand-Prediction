
import os
import joblib
import mlflow
import mlflow.pyfunc
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

MODEL_PATH = "E:\\Depi\\final project\\docker files (1)\\models\\rossmann_sales_model.pkl"  

MODEL_PARAMS = {                         
    "n_estimators": 400,
    "max_depth": 22,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
}

MODEL_METRICS = {                      
    "rmse": 987.696,
    "mae": 668.378,
    "mape": 19.918,
    "mse": 975543.148,
    "R2": 0.899,
}

EXPERIMENT_NAME = "sales-forecasting"
REGISTERED_MODEL_NAME = "sales-forecast-model"


class SalesForecastWrapper(mlflow.pyfunc.PythonModel):
    """Wraps model + scaler + feature_cols together so MLflow can serve
    them as a single unit with the standard python_function flavor."""

    def load_context(self, context):
        bundle = joblib.load(context.artifacts["model_bundle"])
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_cols"]

    def predict(self, context, model_input):
        X = model_input[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="initial-model-registration") as run:
    for key, value in MODEL_PARAMS.items():
        mlflow.log_param(key, value)

    for key, value in MODEL_METRICS.items():
        mlflow.log_metric(key, value)

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SalesForecastWrapper(),
        artifacts={"model_bundle": MODEL_PATH},
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Model registered as '{REGISTERED_MODEL_NAME}'")
    print("Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1' and open http://localhost:5000 to view it.")
