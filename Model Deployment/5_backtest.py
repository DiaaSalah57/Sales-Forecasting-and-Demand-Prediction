"""
STEP 4b: Backtest -- populate monitoring logs with real held-out data.

This takes rows from your actual time-series validation set (features +
real Sales), sends each row's features through the live API to get a
prediction, and logs BOTH the prediction and the real actual sales value
under the same timestamp. This gives 4_monitor.py real data to compare
against instead of made-up numbers.

PREREQUISITE: the API must already be running in another terminal:
    uvicorn 2_app:app --reload --port 8000

Run this with:
    python 5_backtest.py
"""

import json
import os
import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VALIDATION_FILE = BASE_DIR / "validation_with_actuals.csv"
API_URL = "http://localhost:8000/predict"

PREDICTIONS_LOG = BASE_DIR / "logs" / "predictions_log.jsonl"
ACTUALS_FILE = BASE_DIR / "logs" / "actual_sales.csv"

N_SAMPLES = 300  # how many rows to backtest -- spread evenly across the full date range

FEATURE_COLS = [
    "Promo", "Promo2Active", "StateHoliday", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpen",
    "DayOfWeek_2", "DayOfWeek_3", "DayOfWeek_4", "DayOfWeek_5", "DayOfWeek_6", "DayOfWeek_7",
    "Month", "Quarter", "Month_sin", "Month_cos", "DOW_sin", "DOW_cos", "IsWeekend",
    "StoreType_b", "StoreType_c", "StoreType_d", "Assortment_b", "Assortment_c",
    "Sales_Lag_7", "Sales_Lag_14", "Sales_Lag_30",
    "Sales_RollingMean_7", "Sales_RollingMean_30", "Sales_RollingStd_7",
    "Store_Month_AvgSales",
]

BOOL_COLS = {
    "Promo", "Promo2Active", "StateHoliday", "SchoolHoliday", "CompetitionOpen", "IsWeekend",
    "DayOfWeek_2", "DayOfWeek_3", "DayOfWeek_4", "DayOfWeek_5", "DayOfWeek_6", "DayOfWeek_7",
    "StoreType_b", "StoreType_c", "StoreType_d", "Assortment_b", "Assortment_c",
}


def main():
    df = pd.read_csv(VALIDATION_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Sample N rows spread evenly across the whole dataset rather than just the start
    step = max(len(df) // N_SAMPLES, 1)
    sample = df.iloc[::step].head(N_SAMPLES).copy()
    print(f"Backtesting {len(sample)} rows out of {len(df)} total...")

    os.makedirs(BASE_DIR / "logs", exist_ok=True)

    # Warm-up check: make sure the API is actually up before firing 300 requests at it
    try:
        health = requests.get("http://localhost:8000/", timeout=5)
        health.raise_for_status()
        print("API is up. Starting backtest...")
    except Exception as e:
        print(f"ERROR: Could not reach the API at all: {e}")
        print("Make sure it's running with: uvicorn 2_app:app --reload --port 8000")
        return

    predictions_out = []
    actuals_out = []
    failures = 0

    for _, row in sample.iterrows():
        # Build the payload, sending real booleans for flag columns
        payload = {}
        for col in FEATURE_COLS:
            val = row[col]
            if col in BOOL_COLS:
                val = bool(val)
            else:
                val = float(val) if isinstance(val, (int, float)) else val
            payload[col] = val

        timestamp = row["Date"].isoformat()

        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            prediction = response.json()["forecast"]
        except Exception as e:
            failures += 1
            print(f"  Skipped one row -- API call failed: {e}")
            continue

        predictions_out.append({
            "timestamp": timestamp,
            "input": payload,
            "prediction": prediction,
        })
        actuals_out.append({
            "timestamp": timestamp,
            "actual_sales": row["Sales"],
        })

    # Append predictions to the existing JSONL log
    with open(PREDICTIONS_LOG, "a") as f:
        for entry in predictions_out:
            f.write(json.dumps(entry) + "\n")

    # Write/overwrite the actuals CSV with these backtested rows
    actuals_df = pd.DataFrame(actuals_out)
    actuals_df.to_csv(ACTUALS_FILE, index=False, encoding="utf-8")

    print(f"Done. {len(predictions_out)} predictions logged, {failures} rows failed.")
    print(f"Predictions: {PREDICTIONS_LOG}")
    print(f"Actuals:     {ACTUALS_FILE}")
    print("Now run: python 4_monitor.py")


if __name__ == "__main__":
    main()