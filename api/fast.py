from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from grid_intelligence.interface.main import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Grid Intelligence API is running!"}

@app.get("/predict")
def get_predict():
    """
    Predict electricity prices for the next 72 hours (288 intervals).
    Uses multi-regime XGBoost with iterative multi-step forecasting.
    """
    result = predict()
    return result
