from fastapi import UploadFile
import torch
import torchaudio

from api.config import settings
from api.services.audio_service import prepare_audio
from api.models.base import BaseDetector
from api.feature_extractors.factory import get_extractor


async def predict_audio(file: UploadFile, detector: BaseDetector):
    """
    Orchestrates the full inference pipeline for a single audio upload.

    The process includes:
    1. Saving and converting the uploaded file to a temporary WAV.
    2. Routing logic:
        - For 'wavlm' models: Performs raw waveform loading, resampling, and mono-conversion.
        - For other models: Uses the configured feature extractor (e.g., Mel-spectrogram).
    3. Executing the model prediction with a defined threshold.

    Args:
        file (UploadFile): The multipart audio file from the request.
        detector (BaseDetector): The loaded ML model instance.

    Returns:
        dict: The prediction results including label and confidence.
    """
    # Step 1: Standardize input format (handled by audio_service)
    wav_path = await prepare_audio(file)

    # Step 2: Extract features based on the specific model requirements
    if settings.DETECTOR.endswith("wavlm"):
        # WavLM-based models usually expect raw 16kHz mono waveforms
        waveform, sr = torchaudio.load(wav_path)

        # Ensure correct sampling rate
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Convert stereo to mono by averaging channels if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        features = waveform
    else:
        # Classical feature-based pipeline (e.g., Spectrograms, LFCC)
        extractor = get_extractor(settings.EXTRACTOR)
        features = extractor.extract(wav_path)

    # Step 3: Run Inference
    inference = detector.predict(features)

    return inference
