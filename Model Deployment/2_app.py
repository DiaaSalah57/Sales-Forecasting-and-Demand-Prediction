#STEP 2: Serve the model as a real API.


from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime

app = FastAPI(title="Sales Forecast API")


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

REGISTERED_MODEL_NAME = "sales-forecast-model"
MODEL_STAGE_OR_VERSION = "latest"  

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE_OR_VERSION}"
)

LOG_FILE = str(BASE_DIR / "logs" / "predictions_log.jsonl")
os.makedirs(BASE_DIR / "logs", exist_ok=True)


class ForecastRequest(BaseModel):
    Promo: int
    Promo2Active: int
    StateHoliday: int
    SchoolHoliday: int
    CompetitionDistance: float
    CompetitionOpen: int
    DayOfWeek_2: int
    DayOfWeek_3: int
    DayOfWeek_4: int
    DayOfWeek_5: int
    DayOfWeek_6: int
    DayOfWeek_7: int
    Month: int
    Quarter: int
    Month_sin: float
    Month_cos: float
    DOW_sin: float
    DOW_cos: float
    IsWeekend: int
    StoreType_b: int
    StoreType_c: int
    StoreType_d: int
    Assortment_b: int
    Assortment_c: int
    Sales_Lag_7: float
    Sales_Lag_14: float
    Sales_Lag_30: float
    Sales_RollingMean_7: float
    Sales_RollingMean_30: float
    Sales_RollingStd_7: float
    Store_Month_AvgSales: float
# --------------------------------------------------------------------


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sales Forecast API is running"}


@app.post("/predict")
def predict(request: ForecastRequest):
    input_df = pd.DataFrame([request.dict()])

    prediction = model.predict(input_df)
    forecast_value = float(prediction[0])

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": request.dict(),
        "prediction": forecast_value,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"forecast": forecast_value}
