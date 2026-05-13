from pathlib import Path

from api.config import settings
from api.models.gcnn import GCNNDetector
from api.models.gwavlm import GNNWavLMDetector
from api.models.mbwavlm import MultiBranchWavLMDetector
from api.models.rcnn import RCNNDetector
from api.models.rwavlm import LSTMWavLMDetector
from api.models.wavlm import WavLMDetector

# Resolve the project root and weights directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"


def load_detector():
    """
    Factory function to initialize and load a spoofing detector based on the environment configuration.

    This function:
    1. Maps the 'DETECTOR' config string to a specific class implementation.
    2. Constructs the expected path for the model checkpoint (.ckpt).
    3. Instantiates the model, loads its weights, and sets it to evaluation mode.

    Returns:
        BaseDetector: An initialized instance of a detector subclass.

    Raises:
        ValueError: If the 'DETECTOR' variable refers to an unregistered model name.
        FileNotFoundError: If the corresponding .ckpt file is missing from the weights directory.
    """
    MODELS_MAP = {
        "rcnn": RCNNDetector,
        "gcnn": GCNNDetector,
        "wavlm": WavLMDetector,
        "gwavlm": GNNWavLMDetector,
        "rwavlm": LSTMWavLMDetector,
        "mbwavlm": MultiBranchWavLMDetector,
    }

    if settings.DETECTOR not in MODELS_MAP:
        raise ValueError(
            f"Unknown detector model '{settings.DETECTOR}'. "
            f"Available options are: {list(MODELS_MAP.keys())}"
        )

    # Standardized naming convention: weights folder must contain {DETECTOR_NAME}.ckpt
    checkpoint_path = WEIGHTS_DIR / f"{settings.DETECTOR}.ckpt"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    model_class = MODELS_MAP[settings.DETECTOR]
    print(f"Initializing detector: {model_class.__name__}")

    # Initialize model instance with the path to its specific checkpoint
    model = model_class(checkpoint_path=str(checkpoint_path))
    
    # Load weights into architecture
    model.load()
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    return model