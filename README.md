```markdown
# Grid Intelligence ⚡

Day-ahead electricity price prediction for the DE-LU bidding zone (Germany/Luxembourg).

Built with a Transformer model (PyTorch) and an XGBoost spike detector, served via FastAPI on Google Cloud Run.

---

## Team

| Member | Role |
|--------|------|
| Kim | Feature Engineering, XGBoost |
| Susanta | Deep Learning, Transformer Model, LSTM Model |
| Laurenz | Baseline Models, SARIMAX |
| Javier | Data Pipeline, MLOps, API, GCP Deployment |

---

## Project Structure

```
grid-intelligence/
├── api/
│   ├── __init__.py
│   └── fast.py                  ← FastAPI endpoints
├── grid_intelligence/
│   ├── data/
│   │   └── fetcher.py           ← Data pipeline (ENTSO-E, Weather, Gas)
│   ├── interface/
│   │   └── main.py              ← Predict function
│   ├── logic/                   ← Model logic
│   ├── params.py                ← Global parameters & constants
│   └── utils.py
├── models/                      ← Saved model files (.pth, .pkl)
├── notebooks/                   ← Jupyter notebooks per team member
│   ├── javier/
│   ├── kim/
│   ├── laurenz/
│   └── susanta/
├── raw_data/                    ← Local CSV data (gitignored)
├── Dockerfile
├── Makefile
├── requirements.txt
├── setup.py
├── .env                         ← Local environment variables (gitignored)
└── .env.yaml                    ← GCP environment variables (gitignored)
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source file | `raw_data/consolidated_full.csv` |
| BigQuery table | `grid-intelligence-2026.grid_intelligence.consolidated` |
| Granularity | 15 minutes |
| Range | 2018-10-01 to present (+3 days forecast) |
| Columns | 25 |
| Timezone | UTC |

### Data Sources

| Source | Data | Library |
|--------|------|---------|
| ENTSO-E | Prices, Generation, Load, Wind | `entsoe-py` |
| Open-Meteo ERA5 | Observed historical weather | `openmeteo-requests` |
| Open-Meteo Forecast | Weather forecast (+3 days) | `openmeteo-requests` |
| Yahoo Finance | TTF, WTI, Brent, Henry Hub | `yfinance` |

### Column Overview

**Target**
- `price` — Day-ahead electricity price in EUR/MWh

**Grid (ENTSO-E)**
- `generation`, `generation_renewable`, `generation_non_renewable`, `consumption`, `wind_onshore`

**Weather — Observed (ERA5)**
- `temperature_c_observed`, `humidity_percent_observed`, `cloud_cover_percent_observed`, `shortwave_radiation_wm2_observed`, `wind_speed_ms_observed`

**Weather — Forecast**
- `temperature_c_forecast`, `humidity_percent_forecast`, `cloud_cover_percent_forecast`, `shortwave_radiation_wm2_forecast`, `wind_speed_ms_forecast`

**Commodities**
- `ttf_gas`, `wti_oil`, `brent_oil`, `natural_gas`

> Full column descriptions: see `notebooks/javier/fetch_description.md`

---

## Local Setup

### 1. Clone the repo
```bash
git clone git@github.com:xucenying/grid-intelligence.git
cd grid-intelligence
git checkout deploy
```

### 2. Create Python environment
```bash
pyenv virtualenv 3.10.6 grid-intelligence
pyenv activate grid-intelligence
```

### 3. Install dependencies
```bash
make install_requirements
make reload
```

### 4. Create `.env` file
```bash
cp .env.example .env
```

Fill in your `.env`:
```dotenv
ENTSOE_API_KEY=your_key_here
DOCKER_IMAGE_NAME=grid-intelligence-api
DOCKER_LOCAL_PORT=8080
GCP_PROJECT=grid-intelligence-2026
GCP_REGION=europe-west1
DOCKER_REPO_NAME=grid-intelligence
GAR_MEMORY=2Gi
START_DATE=2018-10-01
END_DATE=2026-04-24
ENV=development
GOOGLE_APPLICATION_CREDENTIALS=/path/to/grid-intelligence-key.json
```

### 5. Create `.env.yaml` for GCP deployment
```yaml
ENTSOE_API_KEY: "your_key_here"
GCP_PROJECT: "grid-intelligence-2026"
GCP_REGION: "europe-west1"
ENV: "production"
```

---

## GCP Authentication

### Login
```bash
gcloud auth login
gcloud config set project grid-intelligence-2026
```

### Application Default Credentials (for BigQuery local access)
```bash
BROWSER="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/bigquery \
  --project=grid-intelligence-2026

gcloud auth application-default set-quota-project grid-intelligence-2026
```

### Service Account Key (alternative)
```bash
# Create service account
gcloud iam service-accounts create grid-intelligence-sa \
    --display-name="Grid Intelligence SA" \
    --project=grid-intelligence-2026

# Grant BigQuery permissions
gcloud projects add-iam-policy-binding grid-intelligence-2026 \
    --member="serviceAccount:grid-intelligence-sa@grid-intelligence-2026.iam.gserviceaccount.com" \
    --role="roles/bigquery.admin"

# Download key
gcloud iam service-accounts keys create ~/.config/gcloud/grid-intelligence-key.json \
    --iam-account=grid-intelligence-sa@grid-intelligence-2026.iam.gserviceaccount.com

# Set in shell
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/grid-intelligence-key.json
```

---

## Data Pipeline

### Delta fetch (daily update)
```bash
make fetch-delta
```
Fetches last 7 days + 3 days forecast, merges with existing data, deduplicates, saves.

### Full fetch (first time only)
```bash
make fetch-full
```

### Production delta fetch (BigQuery)
```bash
ENV=production make fetch-delta
```

### Upload CSV to BigQuery (one-time)
```bash
bq load \
  --source_format=CSV \
  --autodetect \
  --skip_leading_rows=1 \
  grid-intelligence-2026:grid_intelligence.consolidated \
  raw_data/consolidated_full.csv
```

### Verify BigQuery data
```bash
bq query --nouse_legacy_sql \
  'SELECT COUNT(*) as total, MIN(datetime_utc) as von, MAX(datetime_utc) as bis
   FROM `grid-intelligence-2026.grid_intelligence.consolidated`'
```

---

## API

### Run locally
```bash
make run_api
```
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict?date=YYYY-MM-DD` | GET | Price prediction for given date |
| `/data?n=10` | GET | Last N rows from dataset |
| `/features` | GET | Last 672 rows (168h) for model input |
| `/fetch-delta` | GET | Trigger delta fetch |

### Production URL
```
https://grid-intelligence-api-824012305183.europe-west1.run.app
```

---

## Docker

### Build and run locally
```bash
make docker_build_local
make docker_run_local
```

### Deploy to GCP
```bash
make deploy
```
Runs: `docker_build` → `docker_push` → `docker_deploy`

---

## GCP Setup (one-time)

### Set project
```bash
gcloud config set project grid-intelligence-2026
make gcloud-set-project
```

### Create Artifact Registry repository
```bash
make docker_create_repo
```

### Configure Docker auth
```bash
make docker_allow
```

### Create BigQuery dataset
```bash
bq mk --dataset --location=europe-west1 grid-intelligence-2026:grid_intelligence
```

### Grant BigQuery permissions to Cloud Run service account
```bash
PROJECT_NUMBER=$(gcloud projects describe grid-intelligence-2026 --format="value(projectNumber)")

gcloud projects add-iam-policy-binding grid-intelligence-2026 \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/bigquery.admin"
```

---

## Cloud Scheduler (automated daily fetch)

### Create daily cron job
```bash
gcloud scheduler jobs create http fetch-delta-daily \
    --schedule="0 6 * * *" \
    --uri="https://grid-intelligence-api-824012305183.europe-west1.run.app/fetch-delta" \
    --http-method=GET \
    --location=europe-west1 \
    --time-zone="Europe/Berlin"
```
Triggers `/fetch-delta` every day at **06:00 Berlin time**.

### Trigger manually
```bash
gcloud scheduler jobs run fetch-delta-daily --location=europe-west1
```

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install package |
| `make install_requirements` | Install all dependencies |
| `make reload` | Editable install |
| `make clean` | Remove build artifacts |
| `make run_api` | Start API locally |
| `make fetch-delta` | Run delta fetch |
| `make fetch-full` | Run full fetch |
| `make docker_build_local` | Build Docker image locally |
| `make docker_run_local` | Run Docker container locally |
| `make docker_build` | Build for GCP (linux/amd64) |
| `make docker_push` | Push image to Artifact Registry |
| `make docker_deploy` | Deploy to Cloud Run |
| `make deploy` | Build + push + deploy |

---

## Known Issues & Notes

- `generation` lags a few hours behind — ENTSO-E publishes actual data with delay
- Legacy weather columns (`temperature_c`, etc.) are past forecast — use `_observed` for training
- Future rows (+3 days): only `_forecast` weather and commodity columns populated
- Python 3.10 shows FutureWarning from `google.api_core` — not critical
- BigQuery Storage module not installed — data fetched via REST (slower but functional)
- `ENV=development` uses CSV locally, `ENV=production` uses BigQuery

---

## Model Performance

| Model | MAE (EUR/MWh) | Horizon |
|-------|---------------|---------|
| Prophet | 72.70 | 1 year |
| ARIMA | 38.93 | 1h |
| XGBoost (72h) | 39.06 | 72h |
| XGBoost (24h) | 31.31 | 24h |
| Transformer V2 | ~26 | 24h |

---

## Tech Stack

- **Model:** PyTorch (Transformer), XGBoost
- **API:** FastAPI, Uvicorn
- **Data:** entsoe-py, openmeteo-requests, yfinance, pandas
- **Infrastructure:** GCP Cloud Run, BigQuery, Cloud Scheduler, Artifact Registry
- **Environment:** Python 3.10.6, pyenv
- **Repo:** github.com/xucenying/grid-intelligence
```
