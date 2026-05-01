import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path

class BaseFeatureExtractor(ABC):
    
    @abstractmethod
    def extract(self, path: Path) -> np.ndarray:
        pass
