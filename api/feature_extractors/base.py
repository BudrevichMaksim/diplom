import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path

class BaseFeatureExtractor(ABC):
    """
    Abstract base class for audio feature extraction.
    
    This class defines the interface for converting raw audio files into 
    numerical representations (features) suitable for model inference.
    """
    
    @abstractmethod
    def extract(self, path: Path) -> np.ndarray:
        """
        Extracts features from an audio file located at the given path.

        Args:
            path (Path): The filesystem path to the audio file (e.g., .wav, .flac).

        Returns:
            np.ndarray: The extracted features as a NumPy array. 
                        The shape depends on the specific implementation 
                        (e.g., [Time, Frequency] for spectrograms).
        
        Raises:
            FileNotFoundError: If the file at 'path' does not exist.
            ValueError: If the audio format is unsupported or the file is corrupted.
        """
        pass