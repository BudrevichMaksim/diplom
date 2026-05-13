import librosa
from pathlib import Path
from typing import Tuple
import numpy as np

def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file from the disk and resamples it to a standard rate.

    Args:
        path (Path): Path to the audio file.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - audio: The audio signal as a floating-point time series.
            - sr: The sampling rate used during loading (fixed at 16,000 Hz).
    
    Note:
        Librosa automatically normalizes the audio to the range [-1.0, 1.0].
    """
    # Load audio with a fixed sampling rate of 16kHz to ensure 
    # compatibility with downstream ML models.
    audio, sr = librosa.load(path, sr=16_000)
    
    return audio, sr