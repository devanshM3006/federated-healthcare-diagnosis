from fastapi import FastAPI
from disease_registry import DISEASES
from app.schemas.heart import HeartPredictionRequest
from UTILS.preprocessing import preprocess_heart
from UTILS.model_loader import load_model

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Federated Healthcare Diagnosis System API is running!"
    }


@app.get("/models")
def get_models():
    available_models = []

    for disease in DISEASES.values():
        available_models.append(disease["name"])

    return {
        "available_models": available_models
    }
@app.post("/predict/heart")
def predict_heart(request: HeartPredictionRequest):

    tensor = preprocess_heart(request)

    return {
        "tensor_shape": list(tensor.shape)
    }