from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from grid_intelligence.interface.main import predict, _get_models, _get_features
from grid_intelligence.data.fetcher import DataFetcher
from grid_intelligence.logic.preprocessor import generate_features
from grid_intelligence.interface.main import predict_multi_regime
import math
import json
import pandas as pd
import pandas_gbq
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up models and features...")
    _get_models()
    _get_features()
    print("Warmup complete!")
    yield


app = FastAPI(
    title="Grid Intelligence API",
    description="Day-ahead electricity price prediction for the DE-LU market",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)


def df_to_records(df) -> list:
    sample = df.reset_index()
    sample['datetime_utc'] = sample['datetime_utc'].astype(str)
    records = []
    for row in sample.to_dict(orient='records'):
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        records.append(clean)
    return records

@app.get("/")
def root():
    return {"message": "Grid Intelligence API is running!"}

@app.get("/predict")
def get_predict():
    result = predict()
    print("🔥 /predict called from local API")
    return result

@app.get("/energy-mix")
def get_energy_mix(days: int = 7):
    """
    Return renewable vs non-renewable generation and consumption for the last N days.
    """
    try:

        from grid_intelligence.params import GCP_PROJECT, BQ_TABLE_ID

        rows = days * 24 * 4  # 15min intervals
        query = f"""
            SELECT datetime_utc, generation_renewable, generation_non_renewable, consumption, price
            FROM `{BQ_TABLE_ID}`
            WHERE datetime_utc IS NOT NULL
            ORDER BY datetime_utc DESC
            LIMIT {rows}
        """
        df = pandas_gbq.read_gbq(query, project_id=GCP_PROJECT)
        df = df.sort_values('datetime_utc').reset_index(drop=True)
        df['datetime_utc'] = df['datetime_utc'].astype(str)

        payload = {
            "timestamps": df['datetime_utc'].tolist(),
            "generation_renewable": [None if math.isnan(v) or math.isinf(v) else round(float(v), 2) for v in df['generation_renewable']],
            "generation_non_renewable": [None if math.isnan(v) or math.isinf(v) else round(float(v), 2) for v in df['generation_non_renewable']],
            "consumption": [None if math.isnan(v) or math.isinf(v) else round(float(v), 2) for v in df['consumption']],
            "price": [None if math.isnan(v) or math.isinf(v) else round(float(v), 2) for v in df['price']],
        }
        return JSONResponse(content=json.loads(json.dumps(payload, cls=SafeEncoder)))
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})

@app.get("/data")
def get_data(n: int = 10):
    try:
        fetcher = DataFetcher()
        df = fetcher._load(tail=n)
        return JSONResponse(content={
            "rows": len(df),
            "last_date": str(df.index.max()),
            "sample": df_to_records(df)
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})

@app.get("/features")
def get_features():
    try:
        fetcher = DataFetcher()
        df = fetcher._load(tail=672)
        return JSONResponse(content={
            "rows": len(df),
            "from": str(df.index.min()),
            "to": str(df.index.max()),
            "features": df_to_records(df)
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})


@app.get("/backtest")
def get_backtest(days: int = 14):
    """
    Return actual vs predicted prices for the last N days.
    """
    logger.info(f"🔁 /backtest called — days={days}")
    try:
        rows = days * 24 * 4
        df_feat = _get_features()

        drop_cols = ['datetime_utc', 'price', 'target_288', 'regime', 'price_bucket', 'future_timestamp']
        predict_df = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

        predictions = predict_multi_regime(predict_df)

        actual = df_feat['price'].tail(rows).tolist()
        preds = [round(float(p), 2) for p in predictions[-rows:]]

        timestamps = []
        if 'datetime_utc' in df_feat.columns:
            timestamps = df_feat['datetime_utc'].tail(rows).astype(str).tolist()
        else:
            timestamps = [str(i) for i in range(rows)]

        return JSONResponse(content={
            "timestamps": timestamps,
            "actual": actual,
            "predicted": preds
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})


@app.get("/fetch-delta")
def run_fetch_delta():
    try:
        fetcher = DataFetcher()
        fetcher.fetch_delta()
        return {"status": "ok", "message": "Delta fetch completed"}
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})
