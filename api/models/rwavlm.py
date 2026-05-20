import torch

from api.models.base import BaseDetector
from ml.models.rwavlm_detector import LSTMWavLMSpoofDetector


class LSTMWavLMDetector(BaseDetector):
    """
    API adapter for inference execution of the LSTM-WavLM Recurrent Spoof Detector.

    Integrates the hybrid `LSTMWavLMSpoofDetector` network (combining a self-supervised WavLM 
    backbone with a bidirectional LSTM recurrent classifier) into the microservice request serving 
    pipeline. Manages model instantiation, device mapping, and raw signal alignment.
    """

    def load(self) -> None:
        """
        Initializes the model architecture and restores optimized parameter configurations.

        Instantiates the core recurrent self-supervised network with production-tuned 
        hyperparameters, standardizes potential 'model.' tracking state-dictionary prefix 
        anomalies via the inherited base class, and transfers the computation graph onto 
        the designated device (`self.device`).
        """
        model_arch = LSTMWavLMSpoofDetector(
            freeze_ssl=True,
            unfreeze_last_n=0,  # CRITICAL: Set to 0 to safeguard frozen self-supervised WavLM weights
            lstm_hidden=128,    # Scale hidden units to 128 (256 is structurally redundant for 768-dim tokens)
            lstm_layers=1,      # A single bidirectional layer provides sufficient temporal modeling capacity
        )
        
        # Bridge potential checkpoint state dict structure discrepancies from PyTorch Lightning
        self.model = self._load_compat_checkpoint(model_arch)
        self.model.to(self.device)

    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        """
        Sanitizes incoming waveform dimensions before feeding into the sequence encoder.

        Evaluates the tensor footprint of the payload, automatically collapsing redundant 
        channel axes (e.g., transforming [Batch, 1, Samples] down to [Batch, Samples]). This 
        retains native multi-item batch structures for high-throughput production streaming.

        Args:
            features (torch.Tensor): Raw waveform audio data tensor.
                Supported structural layouts:
                - 2D: [Batch, Samples] (Standard pre-batched acoustic layout)
                - 3D: [Batch, 1, Samples] (Mono-channel waveform sequence with explicit channel axis)

        Returns:
            torch.Tensor: Normalized waveform packet of shape [Batch, Samples] mapped 
                directly to the upstream self-supervised backbone.
        """
        # Collapse intermediate channel singletons: [Batch, 1, Samples] -> [Batch, Samples]
        if features.dim() == 3 and features.shape[1] == 1:
            features = features.squeeze(1)
            
        return features