import torch

from api.models.base import BaseDetector
from ml.models.gwavlm_detector import GNNWavLMSpoofDetector


class GNNWavLMDetector(BaseDetector):
    """
    API adapter for inference execution of the Graph Neural Network WavLM Spoof Detector.

    Integrates the hybrid `GNNWavLMSpoofDetector` model (combining a self-supervised WavLM 
    backbone with a GNN-based classifier) into the production serving pipeline. Manages 
    weight loading, device mapping, and raw waveform tensor alignment.
    """

    def load(self) -> None:
        """
        Initializes the model architecture and maps parameters from a saved checkpoint.

        Instantiates the core `GNNWavLMSpoofDetector` graph, normalizes any structural 
        prefix discrepancies using the base class checkpoint alignment utilities, and 
        relocates the model parameters to the designated compute resource (`self.device`).
        """
        model_arch = GNNWavLMSpoofDetector()
        
        # Resolve potential 'model.' prefix positioning anomalies from PyTorch Lightning checkpoints
        self.model = self._load_compat_checkpoint(model_arch)
        self.model.to(self.device)

    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        """
        Prepares raw waveform signals for the WavLM feature extraction phase.

        Sanitizes input tensors by stripping out redundant singleton dimensions (e.g., 
        unintentional channel axes produced by certain audio loaders) and enforces 
        a proper batch axis configuration prior to feeding the sequence into the 
        upstream transformer backbone.

        Args:
            features (torch.Tensor): Input raw audio waveform tensor.
                Supported structural variations:
                - 1D: [Samples] (Raw audio signal stream)
                - 2D: [Channels, Samples] (Standard audio signal structure)
                - 3D: [Batch, 1, Samples] (Acoustic packet containing a singleton channel axis)

        Returns:
            torch.Tensor: Normalized waveform packet ready for downstream SSL extraction.
                Guarantees the appending of a batch axis at dimension 0.
        """
        # Scenario 1: Strip out redundant inner singleton dimensions 
        # Transform: [Batch, 1, Samples] -> [Batch, Samples]
        if features.dim() == 3 and features.shape[1] == 1:
            features = features.squeeze(1)

        # Scenario 2: Enforce batch representation boundary
        # If 1D [Samples] -> Becomes [1, Samples] (Ideal for WavLM expectations)
        # If 2D [Channels, Samples] -> Becomes [1, Channels, Samples]
        return features.unsqueeze(0)