import librosa
from pathlib import Path

import numpy as np

from api.feature_extractors.base import BaseFeatureExtractor


class MelSpectrogramExtractor(BaseFeatureExtractor):
    """
    Extractor class for generating Mel-scaled spectrograms from audio files.
    """

    def extract(self, path: Path) -> np.ndarray:
        """
        Loads an audio file and converts it into a Mel-spectrogram.

        The audio is automatically resampled to 16,000 Hz during loading.

        Args:
            path (Path): Path to the source audio file.

        Returns:
            np.ndarray: Mel-spectrogram array with shape [n_mels, time].
                        By default, this returns 128 Mel bands.
        """
        # Load audio with a fixed sampling rate of 16kHz for consistency
        audio, sr = librosa.load(path, sr=16_000)

        # Compute the Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)

        return mel
