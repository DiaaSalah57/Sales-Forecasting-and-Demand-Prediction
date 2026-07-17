# Sales Forecast MLOps Pipeline

A complete local MLOps setup: experiment tracking, deployment, dashboard, and monitoring.

## What's in this project

| File | Purpose |
|---|---|
| `1_log_model_to_mlflow.py` | Registers your trained model with MLflow for tracking/versioning |
| `2_app.py` | FastAPI service that serves live predictions |
| `3_dashboard.py` | Streamlit dashboard for viewing/requesting forecasts |
| `4_monitor.py` | Compares predictions vs actuals, flags drift |
| `RETRAINING_STRATEGY.md` | Written policy for when/how to retrain |

## One-time setup

```bash
cd sales-forecast-mlops
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put your real trained model file at `models/model.pkl`.

## Running it, in order

**1. Register your model with MLflow**

Edit the params/metrics at the top of `1_log_model_to_mlflow.py` to match your
real results, then:
```bash
python 1_log_model_to_mlflow.py
```
Check it worked:
```bash
mlflow ui
```
Open `http://localhost:5000` — you should see your run with its metrics and the model artifact.

**2. Start the API**

First edit the `ForecastRequest` class in `2_app.py` to list the actual
features your model expects (right now it's a placeholder: store_id, date,
promo, holiday).
```bash
uvicorn 2_app:app --reload --port 8000
```
Open `http://localhost:8000/docs` — you'll get an interactive page where you
can send a test request and see a real prediction come back.

**3. Start the dashboard** (in a new terminal, API must still be running)
```bash
streamlit run 3_dashboard.py
```
Opens automatically at `http://localhost:8501`.

**4. Run monitoring** (once you have real sales figures to compare against)

Create `logs/actual_sales.csv` with columns: `date, store_id, actual_sales`
```bash
python 4_monitor.py
```

## MLOps Report (summary)

- **Experiment tracking:** MLflow logs every training run's parameters,
  metrics (RMSE/MAE/MAPE), and the model artifact itself, giving full
  reproducibility -- any past model version can be reloaded exactly.
- **Deployment:** The model is served via a FastAPI REST endpoint
  (`POST /predict`), decoupling the model from any single script or notebook.
  Every prediction is logged for later evaluation.
- **Dashboard:** Streamlit provides a non-technical UI on top of the API for
  business users to request and view forecasts.
- **Monitoring:** Predictions are logged with timestamps; once ground-truth
  sales are known, `4_monitor.py` computes rolling MAPE and raises an alert
  if error exceeds a defined threshold (default 15%).
- **Retraining:** See `RETRAINING_STRATEGY.md` for the full policy
  (scheduled, performance-triggered, and seasonal retraining).

## Running it with Docker (recommended for deployment)

This spins up three containers together: the MLflow tracking server, the
FastAPI prediction service, and the Streamlit dashboard. They're wired to
talk to each other automatically via `docker-compose.yml`.

**One-time prerequisite:** [install Docker Desktop](https://www.docker.com/products/docker-desktop/)
and make sure it's running.

**1. Register your model first (still done locally, before containerizing)**
```bash
python 1_log_model_to_mlflow.py
```
This creates a local `mlruns/` folder — Docker Compose will pick this up and
share it with the containers via a volume, so your registered model is
available inside them.

**2. Edit the placeholder features**

Same as before — update `ForecastRequest` in `2_app.py` and the input fields
in `3_dashboard.py` to match your model's real features before building.

**3. Build and start everything**
```bash
docker compose up --build
```
That's it. Docker will:
- Build the `api` image and `dashboard` image from their Dockerfiles
- Start the `mlflow-server` container first
- Start `api`, which loads your registered model from the shared MLflow store
- Start `dashboard`, which talks to `api` over Docker's internal network

**4. Access everything**
| Service | URL |
|---|---|
| API docs / test page | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

**5. Stop everything**
```bash
docker compose down
```
(Your MLflow data persists in a Docker volume, so it's still there next time
you run `docker compose up`.)

**Retraining with Docker:** run `1_log_model_to_mlflow.py` again locally (or
inside a container) to register a new version, then restart just the API so
it reloads the "latest" model:
```bash
docker compose restart api
```

## Moving to the cloud later

When you're ready to move off your own machine, the same `2_app.py` and
`3_dashboard.py` run unmodified on services like AWS Elastic Beanstalk, GCP
Cloud Run, or Azure App Service -- you'd just containerize with Docker and
point MLflow's tracking URI at a shared server/database instead of your local
disk. Happy to walk through that when you're at that stage.
