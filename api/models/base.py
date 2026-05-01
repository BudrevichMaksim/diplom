from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        pass
