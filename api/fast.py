from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from grid_intelligence.interface.main import predict
from grid_intelligence.data.fetcher import DataFetcher
import numpy as np
import math
import json

app = FastAPI()

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

def safe_json(data):
    return json.loads(json.dumps(data, cls=SafeEncoder))

def df_to_records(df):
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
def get_predict(date: str = "2026-04-24"):
    result = predict(date)
    return result

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

@app.get("/fetch-delta")
def run_fetch_delta():
    try:
        fetcher = DataFetcher()
        fetcher.fetch_delta()
        return {"status": "ok", "message": "Delta fetch completed"}
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})
