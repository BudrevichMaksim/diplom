from typing import Dict, Union

import numpy as np
import torch

from api.models.base import BaseDetector
from ml.models.wavlm_detector import WavLMSpoofDetector


class WavLMDetector(BaseDetector):
    """
    API adapter for inference execution of the Baseline WavLM Spoof Detector.

    Integrates the primary `WavLMSpoofDetector` network into the production serving 
    infrastructure. Handles weight alignment from PyTorch Lightning checkpoints, execution 
    graph routing to compute hardware, and input waveform tensor transformations.
    """

    def load(self) -> None:
        """
        Initializes the model architecture and loads weights from a compatible checkpoint.

        Instantiates the core `WavLMSpoofDetector` network, leverages the base class utility 
        loaders to reconcile state dictionary prefix anomalies, and assigns the model parameters 
        to the active hardware engine (`self.device`).
        """
        model_arch = WavLMSpoofDetector()
        
        # Adjust potential 'model.' prefix positioning anomalies from PyTorch Lightning checkpoints
        self.model = self._load_compat_checkpoint(model_arch)
        self.model.to(self.device)

    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        """
        Sanitizes and formats raw waveform dimensions before entering the SSL backbone.

        Automatically flattens redundant mono-channel dimensions (e.g., converting 
        [Batch, 1, Samples] down to [Batch, Samples]) and appends an external inference 
        batch wrapper.

        Args:
            features (torch.Tensor): Input raw audio waveform tensor.
                Supported structural variants:
                - 1D: [Samples] (Raw audio track stream)
                - 2D: [Batch, Samples] (Standard audio signal structure)
                - 3D: [Batch, 1, Samples] (Acoustic frame array with an explicit channel axis)

        Returns:
            torch.Tensor: Formatted waveform packet. Note that if a 3D input is passed, 
                this method outputs a shape of [1, Batch, Samples], which is automatically 
                flattened back to 2D [Batch, Samples] by the underlying model's internal 
                `forward` view-reshape layer.
        """
        # Scenario 1: Strip out redundant inner channel dimensions
        # Transform: [Batch, 1, Samples] -> [Batch, Samples]
        if features.dim() == 3 and features.shape[1] == 1:
            features = features.squeeze(1)

        # Scenario 2: Enforce global inference batch wrapping boundary
        # If 1D [Samples] -> Becomes [1, Samples]
        # If 2D [Batch, Samples] -> Becomes [1, Batch, Samples]
        return features.unsqueeze(0)