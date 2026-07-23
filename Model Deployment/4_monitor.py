#STEP 4: Monitor model performance over time.



import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_LOG = BASE_DIR / "logs" / "predictions_log.jsonl"
ACTUALS_FILE = BASE_DIR / "logs" / "actual_sales.csv"   
ALERT_THRESHOLD_MAPE = 0.15                             


def load_predictions():
    rows = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df[["timestamp", "prediction"]]


def compute_drift():
    if not PREDICTIONS_LOG.exists():
        print("No predictions logged yet -- nothing to compare.")
        return
    if not ACTUALS_FILE.exists():
        print(f"No actuals file found at {ACTUALS_FILE}. Create it with columns: timestamp, actual_sales")
        return

    preds = load_predictions()
    try:
        actuals = pd.read_csv(ACTUALS_FILE, parse_dates=["timestamp"], encoding="utf-8-sig")
    except UnicodeDecodeError:
        actuals = pd.read_csv(ACTUALS_FILE, parse_dates=["timestamp"], encoding="cp1252")

    actuals.columns = actuals.columns.str.strip()  


    bad_rows = actuals["timestamp"].isna() | actuals["actual_sales"].isna()
    if bad_rows.any():
        print(f"Warning: dropping {bad_rows.sum()} row(s) with missing/invalid timestamp or actual_sales.")
        actuals = actuals[~bad_rows]

    if actuals.empty:
        print("No valid rows remain in actual_sales.csv after cleaning -- check the file's format.")
        return

    merged = pd.merge_asof(
        actuals.sort_values("timestamp"),
        preds.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
    )

    if merged.empty or merged["prediction"].isna().all():
        print("No matching predictions found yet -- nothing to compare.")
        return

    merged["abs_pct_error"] = (
        (merged["actual_sales"] - merged["prediction"]).abs() / merged["actual_sales"]
    )
    mape = merged["abs_pct_error"].mean()

    print(f"Current MAPE across {len(merged)} predictions: {mape:.2%}")

    if mape > ALERT_THRESHOLD_MAPE:
        send_alert(mape)
    else:
        print("Model performance is within acceptable range.")


def send_alert(mape):
    print("=" * 50)
    print(f"ALERT: Model drift detected. MAPE = {mape:.2%}")
    print(f"   Threshold was {ALERT_THRESHOLD_MAPE:.2%}. Consider retraining.")
    print("=" * 50)
  

if __name__ == "__main__":
    compute_drift()
