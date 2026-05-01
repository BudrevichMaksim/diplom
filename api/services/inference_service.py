from os import getenv

from fastapi import UploadFile

from api.config import DETECTOR, EXTRACTOR
from api.feature_extractors.factory import get_extractor
from api.models.registry import get_detector
from api.services.audio_service import prepare_audio


async def predict_audio(file: UploadFile):
    wav_path = await prepare_audio(file)
    extractor = get_extractor(EXTRACTOR)

    features = extractor.extract(wav_path)

    detector = get_detector(DETECTOR)

    inference = detector.predict(features)

    prediction = inference["prediction"]
    confidence = inference["confidence"]

    return {  
        "prediction": prediction,
        "confidence": confidence
    }
