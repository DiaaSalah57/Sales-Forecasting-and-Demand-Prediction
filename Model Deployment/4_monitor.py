"""
STEP 4: Monitor model performance over time.

The idea: every day/week, once you know the REAL sales figures, compare them
to what the model predicted. If the error grows past a threshold, that's drift.

Your model's inputs don't include a raw "date" or "store_id" (they were
engineered away into lag/rolling features), so this version matches
predictions to actuals by TIMESTAMP instead -- you log the actual sales
figure soon after each prediction is made, tagged with the same timestamp
window.

Run it with:
    python 4_monitor.py

In a real setup, schedule this to run automatically (e.g. cron job, Task
Scheduler, or an Airflow DAG) once new actuals come in.
"""

import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_LOG = BASE_DIR / "logs" / "predictions_log.jsonl"
ACTUALS_FILE = BASE_DIR / "logs" / "actual_sales.csv"   # columns: timestamp, actual_sales
ALERT_THRESHOLD_MAPE = 0.15                              # 15% error triggers an alert -- tune this


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
        # Common when the CSV was saved from Excel with a Windows encoding
        actuals = pd.read_csv(ACTUALS_FILE, parse_dates=["timestamp"], encoding="cp1252")

    actuals.columns = actuals.columns.str.strip()  # remove stray whitespace in header names

    # Drop any rows where the timestamp or actual_sales value didn't parse cleanly
    # (e.g. blank trailing rows, malformed dates) instead of crashing.
    bad_rows = actuals["timestamp"].isna() | actuals["actual_sales"].isna()
    if bad_rows.any():
        print(f"Warning: dropping {bad_rows.sum()} row(s) with missing/invalid timestamp or actual_sales.")
        actuals = actuals[~bad_rows]

    if actuals.empty:
        print("No valid rows remain in actual_sales.csv after cleaning -- check the file's format.")
        return

    # Match each actual to the nearest prediction in time
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
    # Replace this with a real notification: email, Slack webhook, etc.
    print("=" * 50)
    print(f"ALERT: Model drift detected. MAPE = {mape:.2%}")
    print(f"   Threshold was {ALERT_THRESHOLD_MAPE:.2%}. Consider retraining.")
    print("=" * 50)
    # Example Slack webhook (uncomment and fill in your webhook URL):
    # import requests
    # requests.post(
    #     "https://hooks.slack.com/services/XXXX/XXXX/XXXX",
    #     json={"text": f"Sales model MAPE hit {mape:.2%} -- retraining recommended."}
    # )


if __name__ == "__main__":
    compute_drift()