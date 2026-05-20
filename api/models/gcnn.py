import torch

from api.models.base import BaseDetector
from ml.models.gcnn_model import GCNNSpoofDetector


class GCNNDetector(BaseDetector):
    """
    API adapter for inference execution of the Graph-Convolutional Spoof Detector.

    Integrates the underlying `GCNNSpoofDetector` ML topology into the microservice 
    request processing workflow. Handles weight alignment, execution graph allocation 
    to the target compute device, and input acoustic matrix batch normalization.
    """

    def load(self) -> None:
        """
        Initializes the structural architecture and loads weights from a compatible checkpoint.

        Instantiates the `GCNNSpoofDetector` network, maps mismatched state dictionary keys 
        using the inherited base adapter utilities, and relocates model parameters to the 
        active execution runtime (`self.device`).
        """
        model_arch = GCNNSpoofDetector()
        
        # Resolve potential 'model.' prefix positioning anomalies from PyTorch Lightning checkpoints
        self.model = self._load_compat_checkpoint(model_arch)
        self.model.to(self.device)

    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        """
        Standardizes variable-dimensional input features into a valid 4D spatial tensor.

        Guarantees that the convolutional encoding layer receives a strictly bound batch shape, 
        safely processing single-channel matrices, multi-channel arrays, or pre-batched streams.

        Args:
            features (torch.Tensor): Input spectrogram maps or raw feature representations.
                Supported structural variations:
                - 2D: [Height, Width] (Single-channel frame sequence without batch axis)
                - 3D: [Channels, Height, Width] (Multi-channel frame sequence without batch axis)
                - 4D: [Batch, Channels, Height, Width] (Valid pre-formatted batch state)

        Returns:
            torch.Tensor: Normalized feature tensor formatted as [Batch, Channels, Height, Width].
        """
        # Scenario 1: Single matrix [H, W] -> Append batch and channel dimensions -> [1, 1, H, W]
        if features.dim() == 2:
            return features.unsqueeze(0).unsqueeze(0)
            
        # Scenario 2: Multi-channel token [C, H, W] -> Append batch dimension only -> [1, C, H, W]
        elif features.dim() == 3:
            return features.unsqueeze(0)
            
        # Scenario 3: Pre-structured data packet [B, C, H, W] -> Forward unmodified
        return features