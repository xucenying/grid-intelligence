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
def get_predict(date: str = "2026-04-24"):
    result = predict(date)
    return result
