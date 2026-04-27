from fastapi import FastAPI
from grid_intelligence.logic.model_sus import initialize_model, predict

app = FastAPI()

# Initialize model silently (runs once when app starts)
initialize_model()


@app.get("/predict")
def get_prediction(steps: int = 3):
    return predict(steps=steps)

@app.get("/")
def root():
    return {"message": "API is working 🚀"}
