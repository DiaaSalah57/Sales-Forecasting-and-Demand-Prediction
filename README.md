# Sales Forecasting and Demand Prediction

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-189AB4)
![MLflow](https://img.shields.io/badge/MLflow-3.14.0-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139.0-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59.1-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> 🚀 **End-to-End Data Science Project**: EDA → Feature Engineering → Model Comparison → MLflow Registry → FastAPI Service → Streamlit Dashboard → Drift Monitoring  
> 📦 **Deployment-Ready**: Dockerized API, dashboard and MLflow tracking server via `docker-compose`  
> 🔧 **Quick Start**: See [Quick Start](#-quick-start) | [Milestone Reports](#-documentation) | [▶️ Dashboard Demo Video](Dashbooard_2.mp4)

Forecast **daily store-level sales** for the Rossmann drugstore chain (1,115 stores, Jan 2013 – Jul 2015) using the Kaggle *Rossmann Store Sales* dataset. The forecasts support inventory planning, marketing and business decisions. Built as part of the **AI & Data Science Track** in three milestones plus a deployment stage.

## 📁 Project Structure

```
Sales-Forecasting-and-Demand-Prediction/
├── Milstone 1/                                  # Milestone 1 - EDA & Preprocessing (folder name spelled "Milstone" in the repo)
│   ├── Milestone_1_Last_updated_clean.ipynb     # Cleaning, outlier treatment, lags, encoding, scaling
│   └── EDA_Report_Milestone_1.pdf               # 7-page report
│
├── Milestone 2/                                 # Milestone 2 - Advanced Analysis & Feature Engineering
│   ├── Milestone2_Advanced_Analysis_Feature_Engineering__LastEdited (1).ipynb
│   └── Milestone2_Report (1).pdf                # 13-page report
│
├── Milestone 3/                                 # Milestone 3 - Modeling (LR / RF / XGBoost / SARIMA)
│   ├── Milestone3_Rossmann_LastEdited.ipynb     # Training, tuning, evaluation, SHAP, model export
│   ├── Milestone3_ReportLastEdited1- (1).pdf    # 16-page report
│   └── 1_log_model_to_mlflow.py                 # Earlier draft of the MLflow registration script
│
├── Model Deployment/                            # Serving, dashboard, monitoring
│   ├── 1_log_model_to_mlflow.py                 # Step 1: register model in MLflow (pyfunc wrapper)
│   ├── 2_app.py                                 # Step 2: FastAPI prediction service
│   ├── 3_dashboard.py                           # Step 3: Streamlit dashboard
│   ├── 4_monitor.py                             # Step 4: MAPE drift monitor + alert
│   ├── 5_backtest.py                            # Step 4b: replay held-out data through the API
│   ├── Dockerfile.api                           # API image (python:3.11-slim)
│   ├── Dockerfile.dashboard                     # Dashboard image (python:3.11-slim)
│   ├── docker-compose.yml                       # mlflow-server + api + dashboard
│   ├── requirements.txt                         # Pinned runtime dependencies
│   ├── requirements-locked.txt                  # Full environment freeze (UTF-16)
│   ├── mlflow.db                                # MLflow SQLite tracking/registry DB
│   ├── validation_with_actuals.csv              # 140,732 held-out rows (45.8 MB) with true Sales
│   └── logs/
│       ├── predictions_log.jsonl                # Every prediction served by the API
│       └── actual_sales.csv                     # 300 actuals written by the backtest
│
├── Dashbooard_2.mp4                             # 60-second Streamlit dashboard demo
└── README.md
```

## 📊 Dataset at a Glance

| Item | Value |
|------|-------|
| **Source** | Rossmann Store Sales (Kaggle): `train.csv` + `store.csv` |
| **Granularity** | One store × one date (store-level, **not** product-level) |
| **Stores / Period** | 1,115 stores · 2013-01-01 → 2015-07-31 |
| **Raw merged data** | 1,017,209 rows × 18 columns |
| **After cleaning** (open stores only) | 844,392 rows × 39 columns |
| **After Milestone 2 feature engineering** | 844,392 rows × 50 columns |
| **Features used for modeling** | 31 (`Customers` deliberately excluded, not known at prediction time) |
| **Train / Test cutoff** | 2015-02-28 → 703,660 train rows / 140,732 test rows |

> ℹ️ The source data has no product/SKU, weather or macro-economic fields. The project therefore forecasts **aggregate daily store-level demand** (documented as *Option A* in the Milestone 1 report).

## 🎯 Models Implemented

### Model 1: Linear Regression (Baseline)
- **Test R²**: 82.27% (Train 0.823 / Test 0.823 → gap 0.001)
- **RMSE**: 1,308.19 · **MAE**: 930.99 · **MAPE**: 22.99%
- **Best for**: Fast, interpretable baseline (no overfitting, limited by linearity)

### Model 2: Random Forest ⭐ Selected for Deployment
- **Test R²**: 88.09% (Train 0.900 / Test 0.881 → gap 0.019)
- **RMSE**: 1,072.29 · **MAE**: 725.94 · **MAPE**: 20.63%
- **Aggregated to daily totals**: R² 93.76%, MAPE 6.23%
- **Best for**: Stable, explainable production model with a small train/test gap

### Model 3: XGBoost
- **Test R²**: 88.58% (Train 0.916 / Test 0.886 → gap 0.030)
- **RMSE**: 1,049.77 · **MAE**: 720.68 · **MAPE**: 20.52%
- **Best for**: Highest raw accuracy on store-level rows

### Statistical Baseline: SARIMA (2,1,2) × (1,1,1,7)
- Fit on the network-wide **daily total** series only (788 train days / 154 test days, AIC 24,590.8)
- **R²**: 63.16% · **MAPE**: 107.71%: clearly outperformed by the feature-rich ML models

## 🚀 Quick Start

### 0. Clone and Install

```bash
git clone https://github.com/DiaaSalah57/Sales-Forecasting-and-Demand-Prediction.git
cd Sales-Forecasting-and-Demand-Prediction

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r "Model Deployment/requirements.txt"

# Extra packages used only by the notebooks
pip install seaborn statsmodels xgboost shap plotly
```

### 1. Explore the Notebooks (run in order)

| # | Notebook | Reads | Produces |
|---|----------|-------|----------|
| 1 | `Milstone 1/Milestone_1_Last_updated_clean.ipynb` | Kaggle `rossmann-store-sales.zip` | `cleaned_rossmann_data.csv` |
| 2 | `Milestone 2/…FeatureEngineering__LastEdited (1).ipynb` | `cleaned_final_rossmann_data.zip` | Engineered dataset + top-15 feature ranking |
| 3 | `Milestone 3/Milestone3_Rossmann_LastEdited.ipynb` | `cleaned_final_rossmann_data_m2.zip` | `rossmann_sales_model.pkl` |

> The notebooks were written in Google Colab (paths such as `/content/...`). Adjust paths if you run them locally, and see [Known Limitations](#-known-limitations--reproducibility-notes) for the data hand-off between notebooks.

### 2. Register the Model in MLflow

Set `MODEL_PATH` in `Model Deployment/1_log_model_to_mlflow.py` to your local `rossmann_sales_model.pkl` (a joblib bundle holding `model`, `scaler` and `feature_cols`), then:

```bash
cd "Model Deployment"
python 1_log_model_to_mlflow.py

# Browse experiments and the model registry
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
# → http://localhost:5000
```

**Expected Output:**
```
Run ID: <run_id>
Model registered as 'sales-forecast-model'
Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1' and open http://localhost:5000 to view it.
```

### 3. Start the Prediction API

```bash
# from inside "Model Deployment"
uvicorn 2_app:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
```

```bash
# Health check
curl http://localhost:8000/
# {"status":"ok","message":"Sales Forecast API is running"}

# Forecast (example: a real validation row, 2015-02-28, actual sales = 5,425)
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "Promo": 0, "Promo2Active": 0, "StateHoliday": 0, "SchoolHoliday": 0,
  "CompetitionDistance": 1.494813, "CompetitionOpen": 124,
  "DayOfWeek_2": 0, "DayOfWeek_3": 0, "DayOfWeek_4": 0, "DayOfWeek_5": 0, "DayOfWeek_6": 1, "DayOfWeek_7": 0,
  "Month": 2, "Quarter": 1, "Month_sin": 0.866025, "Month_cos": 0.5,
  "DOW_sin": -0.781831, "DOW_cos": 0.62349, "IsWeekend": 1,
  "StoreType_b": 0, "StoreType_c": 0, "StoreType_d": 0, "Assortment_b": 0, "Assortment_c": 1,
  "Sales_Lag_7": -0.46806, "Sales_Lag_14": -0.811056, "Sales_Lag_30": 0.520487,
  "Sales_RollingMean_7": -0.333999, "Sales_RollingMean_30": -0.262505, "Sales_RollingStd_7": -0.801581,
  "Store_Month_AvgSales": -0.390199
}'
# → {"forecast": <float>}
```

> ⚠️ The values above are **standardized** (z-scores), exactly as stored in `validation_with_actuals.csv`. See [Known Limitations](#-known-limitations--reproducibility-notes).

### 4. Launch the Dashboard

```bash
# from inside "Model Deployment" (the dashboard reads logs/predictions_log.jsonl)
streamlit run 3_dashboard.py
# → http://localhost:8501
```

### 5. Backtest and Monitor

```bash
# Replays 300 evenly-spaced validation rows through the running API
python 5_backtest.py

# Compares logged predictions with actuals and raises a drift alert when MAPE > 15%
python 4_monitor.py
```

**Sample output of the monitor on the logs shipped in this repo:**
```
Current MAPE across 300 predictions: 27.74%
==================================================
ALERT: Model drift detected. MAPE = 27.74%
   Threshold was 15.00%. Consider retraining.
==================================================
```

## 📊 Performance Comparison

### Store-Level (row-level) Test Set, 140,732 rows

| Metric | Linear Regression | Random Forest ⭐ | XGBoost |
|--------|-------------------|------------------|---------|
| **R² (Accuracy %)** | 82.27 | 88.09 | **88.58** |
| **Train R² / Gap** | 0.823 / 0.001 | 0.900 / **0.019** | 0.916 / 0.030 |
| **RMSE** | 1,308.19 | 1,072.29 | **1,049.77** |
| **MAE** | 930.99 | 725.94 | **720.68** |
| **MSE** | 1,711,352.74 | 1,149,808.46 | **1,102,021.25** |
| **MAPE (%)** | 22.99 | 20.63 | **20.52** |

### Network-Wide Daily Totals, 154 test days

| Metric | SARIMA (2,1,2)×(1,1,1,7) | Random Forest (aggregated) | Improvement |
|--------|--------------------------|----------------------------|-------------|
| **R² (%)** | 63.16 | **93.76** | +30.6 pts |
| **RMSE** | 2,013,772 | **828,758** | **-58.8%** ⚡ |
| **MAE** | 1,318,949 | **426,903** | **-67.6%** ⚡ |
| **MAPE (%)** | 107.71 | **6.23** | **-101.5 pts** ⚡ |

**Key Insight:** Random Forest reaches ~99.4% of XGBoost's R² (88.09 vs 88.58) with a smaller train/test gap (0.019 vs 0.030) and a simpler hyperparameter space, which is why it was chosen for deployment.

## 📖 Documentation

### Milestone Reports (PDF)
- **[Milestone 1: EDA & Preprocessing Report](Milstone%201/EDA_Report_Milestone_1.pdf)**: dataset scoping, data quality, outliers, skewness, feature logic fixes
- **[Milestone 2: Advanced Analysis & Feature Engineering Report](Milestone%202/Milestone2_Report%20%281%29.pdf)**: decomposition, ADF test, correlations, engineered features, 4-method feature selection
- **[Milestone 3: Machine Learning Modeling Report](Milestone%203/Milestone3_ReportLastEdited1-%20%281%29.pdf)**: models, validation strategy, ARIMA comparison, permutation importance, SHAP

### Notebooks
- **[Milestone 1 notebook](Milstone%201/Milestone_1_Last_updated_clean.ipynb)**
- **[Milestone 2 notebook](Milestone%202/Milestone2_Advanced_Analysis_Feature_Engineering__LastEdited%20%281%29.ipynb)**
- **[Milestone 3 notebook](Milestone%203/Milestone3_Rossmann_LastEdited.ipynb)**

### Demo
- **[Dashbooard_2.mp4](Dashbooard_2.mp4)**: 60-second screen recording of the Streamlit dashboard (input form → forecast → predictions log → prediction chart)

## 🏗️ Architecture Details

### Data & Modeling Pipeline

```
Raw Data (Kaggle: train.csv + store.csv)
    • 1,017,209 rows × 18 columns, 1,115 stores
    ↓  left-merge on Store
Milestone 1: Cleaning & Preprocessing
    • Missing-value flags + imputation (structural NaNs → 0 / "None")
    • Drop closed-store rows (Open = 0)            → 844,392 rows
    • IQR capping: Customers, CompetitionDistance  (Sales intentionally NOT capped)
    • Time features, CompetitionOpen, Promo2Active, calendar-based Sales_Lag_7 / _30
    • Encode StateHoliday / StoreType / Assortment / DayOfWeek + StandardScaler
    ↓  844,392 rows × 39 columns
Milestone 2: Advanced Analysis & Feature Engineering
    • Trend / seasonality decomposition, ADF stationarity test, correlations
    • Rolling mean/std (shift(1), no leakage), Sales_Lag_14, cyclical sin/cos encodings
    • Store_Month_AvgSales, 4-method consensus feature selection
    ↓  844,392 rows × 50 columns
Milestone 3: Modeling
    • Chronological split (cutoff 2015-02-28) + TimeSeriesSplit tuning
    • Train-only Store_Month_AvgSales (leakage fix) → StandardScaler
    • Linear Regression · Random Forest · XGBoost · SARIMA
    • Permutation importance + SHAP
    ↓  rossmann_sales_model.pkl  {model, scaler, feature_cols}
Model Deployment
    • MLflow registry → FastAPI → Streamlit → drift monitoring
```

### Serving Architecture

```
┌────────────────────┐   POST /predict   ┌──────────────────────────┐
│ Streamlit Dashboard│ ────────────────▶ │  FastAPI  (2_app.py)     │
│  (3_dashboard.py)  │ ◀──────────────── │  port 8000               │
│  port 8501         │   {"forecast": x} └───────────┬──────────────┘
└─────────▲──────────┘                               │ models:/sales-forecast-model/latest
          │ reads last 20 rows                       ▼
          │                        ┌──────────────────────────────────┐
          │                        │ MLflow pyfunc: SalesForecastWrapper│
          │                        │   scaler.transform → RandomForest │
          │                        └──────────────────────────────────┘
          │                                          │
   logs/predictions_log.jsonl ◀── append (timestamp, input, prediction)
          │
          ▼
   4_monitor.py ── compares with logs/actual_sales.csv ──▶ ALERT if MAPE > 15%
```

### Feature Set (31 features)

| Group | Features |
|-------|----------|
| **Promotions & holidays** | `Promo`, `Promo2Active`, `StateHoliday`, `SchoolHoliday` |
| **Competition** | `CompetitionDistance`, `CompetitionOpen` |
| **Calendar** | `DayOfWeek_2` … `DayOfWeek_7`, `Month`, `Quarter`, `IsWeekend` |
| **Cyclical encodings** | `Month_sin`, `Month_cos`, `DOW_sin`, `DOW_cos` |
| **Store attributes** | `StoreType_b/c/d`, `Assortment_b/c` |
| **Lag features** | `Sales_Lag_7`, `Sales_Lag_14`, `Sales_Lag_30` |
| **Rolling statistics** | `Sales_RollingMean_7`, `Sales_RollingMean_30`, `Sales_RollingStd_7` |
| **Aggregation** | `Store_Month_AvgSales` |

## 🎓 Training Strategy

### Time-Based Validation

```yaml
- Split: Chronological (never random K-Fold)
- TimeSeriesSplit: 5 folds (growing training window)
- Final evaluation fold cutoff: 2015-02-28
- Train rows: 703,660  |  Test rows: 140,732
- Scaling: StandardScaler fit on train only
```

### Data-Leakage Fix (`Store_Month_AvgSales`)

```yaml
- Problem: feature was originally computed on the full dataset
- Fix: recomputed with TRAIN-ONLY statistics
    fallback 1: store's overall training mean
    fallback 2: global training mean
- Effect: value changed in 100% of rows → small, honest drop in reported R²
```

### Hyperparameter Tuning (RandomizedSearchCV, scoring = neg. RMSE, cv = TimeSeriesSplit(2))

**Random Forest**: search on every 5th training row (n_iter = 5), final model refit on the full training set
```yaml
Search space: n_estimators [100,150,200] · max_depth [8,10,12] · min_samples_split [5,10]
              min_samples_leaf [3,5,8] · max_features [sqrt, log2]
Best: n_estimators=200, max_depth=12, min_samples_split=10, min_samples_leaf=3, max_features=sqrt
```

**XGBoost**: search on the full training set (n_iter = 10, `tree_method="hist"`)
```yaml
Search space: n_estimators [300,500,700,900] · max_depth [4,6,8] · learning_rate [0.03…0.15]
              subsample [0.7,0.85,1.0] · colsample_bytree [0.7,0.85,1.0]
              reg_alpha [0,0.1,1] · reg_lambda [1,1.5,2]
Best: n_estimators=900, max_depth=6, learning_rate=0.03, subsample=0.7,
      colsample_bytree=1.0, reg_alpha=0.1, reg_lambda=2
```

**SARIMA**: grid over (p,1,q) with p,q ∈ {0,1,2} and seasonal (P,1,Q,7) with P,Q ∈ {0,1}, selected by AIC → **(2,1,2)×(1,1,1,7)**

### Feature Selection (Milestone 2)

Four independent methods on a 150,000-row sample of 31 candidates, merged into one consensus ranking (lower average rank = stronger):

| Method | Type |
|--------|------|
| Pearson correlation | Filter |
| Mutual information | Filter (non-linear) |
| Random Forest importance (150 trees, depth 12, R² 0.8888 on hold-out) | Embedded |
| Recursive Feature Elimination (Ridge) | Wrapper |

**Top 5 consensus features:** `Sales_Lag_14` (1.50) · `Store_Month_AvgSales` (1.75) · `Sales_RollingMean_7` (4.75) · `Promo` (5.50) · `Sales_Lag_7` (5.75)

> Milestone 2 recommends a top-15 subset; the Milestone 3 models are trained on all 31 candidates.

## 🛠️ Development Tools

### API Reference (`Model Deployment/2_app.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Takes the 31 features (`ForecastRequest`), returns `{"forecast": float}` and appends the request + prediction to `logs/predictions_log.jsonl` |

Environment variable: `MLFLOW_TRACKING_URI` (defaults to the local `mlflow.db` SQLite file).

### Scripts

| Script | Role | Run |
|--------|------|-----|
| `1_log_model_to_mlflow.py` | Logs params/metrics and registers `sales-forecast-model` (experiment `sales-forecasting`) with a `SalesForecastWrapper` that bundles scaler + model + feature list | `python 1_log_model_to_mlflow.py` |
| `2_app.py` | FastAPI service loading `models:/sales-forecast-model/latest` | `uvicorn 2_app:app --port 8000` |
| `3_dashboard.py` | Streamlit form (promotions, store info, date info, recent sales history) + last-20 predictions table + prediction chart | `streamlit run 3_dashboard.py` |
| `4_monitor.py` | Merges predictions with `actual_sales.csv`, prints MAPE, alerts above 15% | `python 4_monitor.py` |
| `5_backtest.py` | Sends 300 evenly spaced validation rows to the API and writes predictions + actuals to `logs/` | `python 5_backtest.py` |

### Dependencies

`requirements.txt` pins: `mlflow==3.14.0` · `fastapi==0.139.0` · `uvicorn==0.51.0` · `streamlit==1.59.1` · `scikit-learn==1.9.0` · `pandas==2.3.3` · `numpy==2.3.5` · `joblib==1.5.3` · `pydantic==2.13.4` · `requests==2.34.2` · `matplotlib==3.11.0`

## 🚀 Deployment Options

### Option 1: Direct Python (local)
```bash
cd "Model Deployment"
python 1_log_model_to_mlflow.py
uvicorn 2_app:app --port 8000          # terminal 1
streamlit run 3_dashboard.py           # terminal 2
```

### Option 2: Docker Compose
```bash
cd "Model Deployment"
docker compose up --build
```

| Service | Image / Dockerfile | Port | Notes |
|---------|--------------------|------|-------|
| `mlflow-server` | `ghcr.io/mlflow/mlflow:latest` | 5000 | SQLite backend, `mlflow-data` volume |
| `api` | `Dockerfile.api` | 8000 | `MLFLOW_TRACKING_URI=http://mlflow-server:5000`, mounts `./logs` |
| `dashboard` | `Dockerfile.dashboard` | 8501 | `API_URL=http://api:8000/predict` |

> `Dockerfile.api` copies a `models/` directory that is **not** in the repository. Create it with your exported artifacts before building.

### Option 3: MLflow UI only
```bash
cd "Model Deployment"
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

## 🎯 Use Cases

| Application | Model Recommendation |
|------------|---------------------|
| **Production deployment (this repo)** | Random Forest (smaller train/test gap, simpler tuning, easier to explain) |
| **Maximum store-level accuracy** | XGBoost (R² 88.58%, RMSE 1,049.77) |
| **Quick interpretable baseline** | Linear Regression (R² 82.27%, no overfitting) |
| **Total daily sales reporting** | Random Forest aggregated (R² 93.76%, MAPE 6.23%) |
| **Explaining sales drivers to stakeholders** | Permutation importance + SHAP |
| **Series with no exogenous features** | SARIMA (weakest here: R² 63.16%) |

## 📈 Expected Results

### Key EDA Findings

- **Promotions:** average sales of 5,929 (no promo) vs 8,228 (promo) → **+38.8% uplift**
- **Weekly seasonality:** Monday is strongest (8,434,351 avg total sales), Sunday weakest (220,533) as most stores close
- **Annual seasonality:** repeatable February dip and a strong run-up into December in all three years
- **Trend:** additive decomposition shows +44.9% (5,279,455 → 7,648,351) across the observed window
- **Stationarity:** ADF p-value = 0.000064 (level) and ≈ 0 (1st difference), so the series is stationary at level
- **Store types:** StoreType `b` has the highest median sales; Assortment `b` ("extra") the highest median
- **Correlation with Sales:** `Customers` 0.819 · `Sales_Lag_7` 0.546 · `Sales_Lag_30` 0.394 · `Promo` 0.368
- **Weak signals:** `CompetitionDistance` (-0.040) and `CompetitionOpen` (-0.003)

### Data-Quality Numbers (Milestone 1)

| Step | Result |
|------|--------|
| Duplicates in `train.csv` / `store.csv` | 0 / 0 |
| Closed-store rows removed | 1,017,209 → 844,392 |
| `Customers` capped (IQR) | 40,853 values (4.84%) → [0, 1,454] |
| `CompetitionDistance` capped (IQR) | 83,041 values (9.83%) → [0, 16,135] |
| Sales skew before → after cleaning | 0.641 → 1.594 (log1p → -0.638) |

### Top Sales Drivers (Milestone 3)

```
Built-in importance (Random Forest):   Sales_Lag_14 0.219 · Store_Month_AvgSales 0.193 ·
                                       Sales_RollingMean_30 0.149 · Sales_RollingMean_7 0.128 · Promo 0.102
Built-in importance (XGBoost):         Sales_Lag_14 0.384 · Store_Month_AvgSales 0.238 · Promo 0.120
Permutation importance (XGBoost):      Store_Month_AvgSales 0.426 · Promo 0.392 · Sales_Lag_14 0.112
Permutation importance (Random Forest):Promo 0.235 · Store_Month_AvgSales 0.119 · Sales_Lag_14 0.090
SHAP (both models):                    Promo, Sales_Lag_14, Store_Month_AvgSales remain the top three
```

### Test Set Performance

```
Random Forest (selected) - 140,732 test rows (2015-02-28 → 2015-07-31):
- R²: 88.09%      RMSE: 1,072.29     MAE: 725.94     MAPE: 20.63%
- Train/Test R² gap: 0.019
- Aggregated to daily totals: R² 93.76%, RMSE 828,758, MAPE 6.23%

Residual analysis:
- Centered around zero (no major systematic bias)
- Variance widens at higher predicted sales (heteroscedasticity)
- Sharp promotion/event spikes tend to be under-predicted
```

## 🔧 Troubleshooting

### Common Issues

**`FileNotFoundError` when registering the model:**
```bash
# MODEL_PATH in 1_log_model_to_mlflow.py is a hard-coded Windows path
# (E:\Depi\final project\docker files (1)\models\rossmann_sales_model.pkl)
# → point it to your own rossmann_sales_model.pkl
```

**API fails to start (model not found):**
```bash
# 2_app.py loads models:/sales-forecast-model/latest
# → run 1_log_model_to_mlflow.py first, from the same folder, so mlflow.db contains a registered model
```

**Backtest prints "Could not reach the API":**
```bash
cd "Model Deployment"
uvicorn 2_app:app --reload --port 8000
```

**Monitor says "No actuals file found":**
```bash
python 5_backtest.py     # creates logs/actual_sales.csv (columns: timestamp, actual_sales)
```

**Dashboard says "Could not reach the API":**
```bash
# The dashboard targets http://localhost:8000/predict by default
export API_URL=http://localhost:8000/predict      # PowerShell: $env:API_URL="http://localhost:8000/predict"
```

**MLflow UI is empty:**
```bash
# Use the same backend store the scripts write to
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

**`NameError: name 'joblib' is not defined` (last cell of the Milestone 3 notebook):**
```python
import joblib   # add to the setup cell
```

**Notebook file-not-found errors:** paths such as `/content/...` come from Google Colab. Change them to your local folders.

## 📊 Monitoring

### What Is Logged

- **`logs/predictions_log.jsonl`**: one JSON line per prediction: `timestamp`, full `input` (31 features), `prediction`
- **`logs/actual_sales.csv`**: `timestamp, actual_sales` written by the backtest
- **MLflow (`mlflow.db`)**: experiment `sales-forecasting` with 9 registration runs (5 finished, 4 failed attempts) and 5 versions of `sales-forecast-model`

### Drift Check

```yaml
- Metric: MAPE = mean(|actual - prediction| / actual)
- Alert threshold: 15%   (ALERT_THRESHOLD_MAPE = 0.15 in 4_monitor.py)
- Matching: nearest timestamp (pd.merge_asof)
- Action on alert: message recommending retraining
```

### MLflow Registration Metadata (as logged by `Model Deployment/1_log_model_to_mlflow.py`)

| Parameter | Value | Metric | Value |
|-----------|-------|--------|-------|
| `n_estimators` | 400 | RMSE | 987.696 |
| `max_depth` | 22 | MAE | 668.378 |
| `min_samples_split` | 5 | MAPE | 19.918 |
| `min_samples_leaf` | 2 | MSE | 975,543.148 |
| `max_features` | sqrt | R² | 0.899 |

## 🚧 Known Limitations & Reproducibility Notes

Please read these before trying to reproduce the pipeline end-to-end.

1. **Artifacts not included in the repo:** the raw Kaggle CSVs, the intermediate cleaned datasets, `rossmann_sales_model.pkl`, the `models/` folder used by `Dockerfile.api`, and the MLflow `mlruns/` artifacts. The shipped `mlflow.db` was created on a Windows machine and points to `file:E:/Depi/final project/docker files (1)/mlruns/...`, so re-register the model locally (Quick Start, step 2) instead of loading the shipped registry.
2. **Notebook hand-offs:** Milestone 1 exports `cleaned_rossmann_data.csv`, while Milestone 2 loads `cleaned_final_rossmann_data.zip`. The Milestone 2 notebook has no cell that exports the 50-column dataset that Milestone 3 loads (`cleaned_final_rossmann_data_m2.zip`). Rename or export accordingly.
3. **Deployed-model metadata vs. notebook run:** the tuned Random Forest in the Milestone 3 notebook uses `n_estimators=200, max_depth=12` and scores R² 88.09% / RMSE 1,072.29, whereas the MLflow registration script records `n_estimators=400, max_depth=22` with R² 0.899 / RMSE 987.70. The training run behind the deployed pickle is not part of the repository.
4. **Feature scale:** the model expects the standardized feature space used in the notebooks (`validation_with_actuals.csv` stores lags, rolling statistics, `CompetitionDistance` and `Store_Month_AvgSales` as z-scores). The dashboard sends **raw-scale** numbers (defaults such as 5,000 and 500). In the shipped log, the 11 dashboard requests with the default `Sales_Lag_7 = 5000` returned forecasts of 21,027 – 24,437 versus a validation-set mean of ≈ 7,204. Also, in the Milestone 3 notebook the leakage-fixed `Store_Month_AvgSales` is stored as raw train-only mean sales. Verify that training features, the scaler and API inputs share one scale before trusting dashboard forecasts.
5. **Monitor accuracy:** `4_monitor.py` matches actuals to predictions by timestamp only (no store key), and `predictions_log.jsonl` mixes manual requests with several appended backtest runs (880 backtest-dated entries vs. 300 actuals). The 27.74% MAPE reported on the shipped logs should therefore be read with caution; matching on a store/row identifier would make it reliable.
6. **Documentation notes:** the Milestone 1 PDF states 38 columns while the notebook output shows 39; the Milestone 2 report lists `Customers` among strong predictors, but Milestone 3 excludes it because it is unknown at prediction time.
7. **`requirements-locked.txt`** is a full-environment freeze saved as UTF-16 and includes Windows-only packages (e.g. `pywin32`); prefer `requirements.txt`.
8. **Naming:** the repository folder is spelled `Milstone 1` and the video `Dashbooard_2.mp4`. Links in this README use the actual names.

## 🤝 Contributing

This is a track project (AI & Data Science Track, Milestones 1–3 + deployment). For questions or improvements:
1. Check the milestone reports in each folder
2. Review the notebook markdown cells and code comments
3. Open an issue or pull request on GitHub

## 📜 License

The code in this repository is released under the [MIT License](LICENSE).

> The Rossmann Store Sales dataset is **not** covered by this license. It is provided by Kaggle/Rossmann under its own terms, so download it from the [competition page](https://www.kaggle.com/c/rossmann-store-sales) and follow its rules.

## 👥 Authors

Contributors (from the Git history):
- **Diaa Salah**: [@DiaaSalah57](https://github.com/DiaaSalah57)
- **EsraaELMahdy**: [@EsraaELMahdy](https://github.com/EsraaELMahdy)

**Date:** July – August 2026  
**Version:** 1.0 (Milestones 1–3 + Model Deployment)

---

## 🎉 Project Highlights

✅ **Full ML lifecycle**: EDA, feature engineering, tuning, evaluation, deployment, monitoring  
✅ **Leakage-aware evaluation**: chronological split + train-only `Store_Month_AvgSales`  
✅ **Three ML models + SARIMA baseline**: compared on both row-level and daily-aggregate scales  
✅ **Explainability**: built-in importance, permutation importance and SHAP (XGBoost & Random Forest)  
✅ **Experiment tracking**: MLflow experiments and model registry  
✅ **Serving stack**: FastAPI + Streamlit + Docker Compose  
✅ **Monitoring**: prediction logging, backtesting and MAPE-based drift alerts  

---

**For detailed analysis, see:**
- [Milestone 1 Report](Milstone%201/EDA_Report_Milestone_1.pdf)
- [Milestone 2 Report](Milestone%202/Milestone2_Report%20%281%29.pdf)
- [Milestone 3 Report](Milestone%203/Milestone3_ReportLastEdited1-%20%281%29.pdf)

**Quick Start:** `cd "Model Deployment"` → `python 1_log_model_to_mlflow.py` → `uvicorn 2_app:app --port 8000` → `streamlit run 3_dashboard.py`
