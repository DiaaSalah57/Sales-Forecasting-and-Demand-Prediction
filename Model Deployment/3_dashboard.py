"""
STEP 3: Interactive dashboard.

Make sure the API (2_app.py) is already running first, then run:
    streamlit run 3_dashboard.py

This opens a browser page at http://localhost:8501

Note: this dashboard exposes every raw feature the model needs, since your
model expects fully engineered inputs (lags, rolling stats, etc.) rather than
simple fields like "date". In a production setup, these would typically be
computed automatically from a database of historical sales -- for now, enter
them directly to test predictions.
"""

import streamlit as st
import requests
import pandas as pd
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Sales Forecast Dashboard", layout="centered")
st.title("📈 Sales Forecast Dashboard")

st.subheader("Get a new forecast")

with st.form("forecast_form"):
    st.markdown("**Promotions & holidays**")
    c1, c2, c3, c4 = st.columns(4)
    promo = c1.checkbox("Promo")
    promo2_active = c2.checkbox("Promo2 Active")
    state_holiday = c3.checkbox("State Holiday")
    school_holiday = c4.checkbox("School Holiday")

    st.markdown("**Store info**")
    c1, c2, c3 = st.columns(3)
    competition_distance = c1.number_input("Competition Distance", value=500.0)
    competition_open = c2.checkbox("Competition Open")
    store_month_avg_sales = c3.number_input("Store-Month Avg Sales", value=5000.0)

    c1, c2, c3 = st.columns(3)
    store_type = c1.selectbox("Store Type", ["a", "b", "c", "d"])
    assortment = c2.selectbox("Assortment", ["a", "b", "c"])
    is_weekend = c3.checkbox("Is Weekend")

    st.markdown("**Date info**")
    c1, c2 = st.columns(2)
    day_of_week = c1.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])
    month = c2.number_input("Month", min_value=1, max_value=12, value=6)
    quarter = (month - 1) // 3 + 1

    st.markdown("**Recent sales history (for this store)**")
    c1, c2, c3 = st.columns(3)
    sales_lag_7 = c1.number_input("Sales 7 days ago", value=5000.0)
    sales_lag_14 = c2.number_input("Sales 14 days ago", value=5000.0)
    sales_lag_30 = c3.number_input("Sales 30 days ago", value=5000.0)

    c1, c2, c3 = st.columns(3)
    rolling_mean_7 = c1.number_input("7-day rolling mean sales", value=5000.0)
    rolling_mean_30 = c2.number_input("30-day rolling mean sales", value=5000.0)
    rolling_std_7 = c3.number_input("7-day rolling std sales", value=500.0)

    submitted = st.form_submit_button("Get Forecast")

if submitted:
    import math

    payload = {
        "Promo": promo,
        "Promo2Active": promo2_active,
        "StateHoliday": state_holiday,
        "SchoolHoliday": school_holiday,
        "CompetitionDistance": competition_distance,
        "CompetitionOpen": competition_open,
        "DayOfWeek_2": day_of_week == 2,
        "DayOfWeek_3": day_of_week == 3,
        "DayOfWeek_4": day_of_week == 4,
        "DayOfWeek_5": day_of_week == 5,
        "DayOfWeek_6": day_of_week == 6,
        "DayOfWeek_7": day_of_week == 7,
        "Month": month,
        "Quarter": quarter,
        "Month_sin": math.sin(2 * math.pi * month / 12),
        "Month_cos": math.cos(2 * math.pi * month / 12),
        "DOW_sin": math.sin(2 * math.pi * day_of_week / 7),
        "DOW_cos": math.cos(2 * math.pi * day_of_week / 7),
        "IsWeekend": is_weekend,
        "StoreType_b": store_type == "b",
        "StoreType_c": store_type == "c",
        "StoreType_d": store_type == "d",
        "Assortment_b": assortment == "b",
        "Assortment_c": assortment == "c",
        "Sales_Lag_7": sales_lag_7,
        "Sales_Lag_14": sales_lag_14,
        "Sales_Lag_30": sales_lag_30,
        "Sales_RollingMean_7": rolling_mean_7,
        "Sales_RollingMean_30": rolling_mean_30,
        "Sales_RollingStd_7": rolling_std_7,
        "Store_Month_AvgSales": store_month_avg_sales,
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        st.success(f"Predicted sales: {result['forecast']:.2f}")
    except Exception as e:
        st.error(f"Could not reach the API. Is it running? Error: {e}")

st.divider()
st.subheader("Recent predictions log")

log_path = "logs/predictions_log.jsonl"
if os.path.exists(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            rows.append(json.loads(line))
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.tail(20))
        st.line_chart(df["prediction"])
    else:
        st.info("No predictions logged yet.")
else:
    st.info("No predictions logged yet. Try getting a forecast above.")