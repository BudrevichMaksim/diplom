import torch

from api.models.base import BaseDetector
from ml.models.mb2wavlm_detector import MultiBranchWavLMSpoofDetector


class MultiBranchWavLMDetector(BaseDetector):
    """
    API adapter for inference execution of the Multi-Branch WavLM Spoof Detector.

    Integrates the state-of-the-art `MultiBranchWavLMSpoofDetector` architecture into the 
    microservice request serving pipeline. Manages automatic weight state-dictionary mapping, 
    compute device localization, and multi-branch tensor normalization.
    """

    def load(self) -> None:
        """
        Initializes the model architecture and restores parameters from a valid checkpoint.

        Instantiates the core multi-branch SSL network structure, standardizes any model 
        prefix anomalies using the inherited base utility loaders, and allocates model parameters 
        to the target processing engine (`self.device`).
        """
        model_arch = MultiBranchWavLMSpoofDetector()
        
        # Adjust potential 'model.' prefix mismatches inherited from PyTorch Lightning checkpoints
        self.model = self._load_compat_checkpoint(model_arch)
        self.model.to(self.device)

    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        """
        Sanitizes incoming waveform dimensions prior to multi-branch feature parsing.

        Evaluates the tensor footprint of the payload, automatically collapsing redundant 
        channel axes (e.g., converting [Batch, 1, Samples] down to [Batch, Samples]). Unlike 
        simpler single-stream baselines, this keeps the original batch mapping intact without 
        introducing artificial single-item grouping axes.

        Args:
            features (torch.Tensor): Raw waveform audio data tensor.
                Expected structural shapes:
                - 2D: [Batch, Samples] (Optimized pre-batched sequence layout)
                - 3D: [Batch, 1, Samples] (Waveform payload with an explicit mono-channel axis)

        Returns:
            torch.Tensor: Normalized waveform packet of shape [Batch, Samples] ready for 
                parallel front-end transformer and prosody extraction layers.
            """
        # Collapse intermediate channel singletons: [Batch, 1, Samples] -> [Batch, Samples]
        if features.dim() == 3 and features.shape[1] == 1:
            features = features.squeeze(1)
            
        return features