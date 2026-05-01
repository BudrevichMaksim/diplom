from fastapi import APIRouter, File, UploadFile

from api.schemas.response import PredictionResponse
from api.services.inference_service import predict_audio

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    return await predict_audio(file) 
