import torch

from api.models.base import BaseDetector
from ml.models.rcnn_model import RCNNSpoofDetector


class RCNNDetector(BaseDetector):
    """
    API adapter for inference execution of the Recurrent Convolutional Spoof Detector.

    Integrates the hybrid `RCNNSpoofDetector` topology (combining 2D spatial feature extraction 
    with recurrent sequential tracking) into the microservice pipeline. Coordinates structural 
    weight loading, target device orchestration, and batch grid alignment.
    """

    def load(self) -> None:
        """
        Initializes the model architecture and restores execution parameters from a checkpoint.

        Instantiates the `RCNNSpoofDetector` network, sanitizes potential 'model.' tracking 
        prefix anomalies via the inherited base class adapter, and maps the computational 
        graph onto the active execution hardware engine (`self.device`).
        """
        model_arch = RCNNSpoofDetector()
        
        # Bridge potential checkpoint state dict structure discrepancies from PyTorch Lightning
        self.model = self._load_compat_checkpoint(model_arch)
        self.model.to(self.device)

    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        """
        Standardizes input acoustic tensors into a uniform 4D spatial footprint.

        Ensures that incoming spectrogram maps (e.g., Mel-spectrograms) are wrapped into a fixed 
        batch structure prior to entering the initial convolutional feature extraction stages. 
        Safely processes isolated frames, multi-channel representations, or pre-aligned batches.

        Args:
            features (torch.Tensor): Input spectrogram representations or acoustic feature maps.
                Supported structural inputs:
                - 2D: [Height, Width] (Single-channel frame matrix without explicit batch grouping)
                - 3D: [Channels, Height, Width] (Multi-channel frame map without explicit batch grouping)
                - 4D: [Batch, Channels, Height, Width] (Valid pre-formatted batch configuration)

        Returns:
            torch.Tensor: Normalized feature tensor formatted as [Batch, Channels, Height, Width].
        """
        # Scenario 1: Unbatched single-channel matrix [H, W] -> Append batch and channel axes -> [1, 1, H, W]
        if features.dim() == 2:
            return features.unsqueeze(0).unsqueeze(0)
            
        # Scenario 2: Unbatched multi-channel slice [C, H, W] -> Append missing batch axis -> [1, C, H, W]
        elif features.dim() == 3:
            return features.unsqueeze(0)
            
        # Scenario 3: Stream is already fully batched [B, C, H, W] -> Pass through unmodified
        return features