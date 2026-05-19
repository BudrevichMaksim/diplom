from fastapi import APIRouter, Depends, File, UploadFile

from shared.schemas.response import PredictionResponse
from api.services.inference_service import predict_audio
from api.dependencies.detectors import get_detector
from api.models.base import BaseDetector

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...), 
    detector: BaseDetector = Depends(get_detector)
) -> PredictionResponse:
    """
    Endpoint to perform spoofing detection on an uploaded audio file.

    This endpoint accepts an audio file via a multipart/form-data request,
    utilizes the injected detector dependency, and returns the classification 
    results (real/fake) with a confidence score.

    Args:
        file (UploadFile): The raw audio file uploaded by the user.
        detector (BaseDetector): The spoofing detector instance, provided via 
        FastAPI dependency injection.

    Returns:
        PredictionResponse: A validated Pydantic model containing the prediction 
                            label and confidence percentage.
    """
    # Delegate the processing, feature extraction, and inference to the service layer
    return await predict_audio(file, detector)