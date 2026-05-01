import librosa
from pathlib import Path

import numpy as np

from api.feature_extractors.base import BaseFeatureExtractor

class MelSpectrogramExtractor(BaseFeatureExtractor):
    
    def extract(self, path: Path) -> np.ndarray:
        audio, sr = librosa.load(path, sr=16_000)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        return mel
