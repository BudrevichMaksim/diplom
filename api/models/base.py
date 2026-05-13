from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F

from ml.lightning_module import SpoofDetectorSystem


class BaseDetector(ABC):
    """
    Abstract base class for spoofing detectors.
    Provides a unified interface for model loading, audio preprocessing, and inference.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initializes the detector with a specific checkpoint.

        Args:
            checkpoint_path (str): Path to the model weights (.ckpt or .pt file).
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """
        Abstract method to initialize the model architecture and load weights.
        Must be implemented by subclasses.
        """
        pass

    def _load_compat_checkpoint(self, model_arch: torch.nn.Module) -> SpoofDetectorSystem:
        """
        Loads a checkpoint with state_dict key mapping for compatibility.
        Ensures that keys match the 'model.layer_name' format used in SpoofDetectorSystem.

        Args:
            model_arch (torch.nn.Module): The backbone model architecture.

        Returns:
            SpoofDetectorSystem: Lightning system with loaded weights.
        """
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]

        # Map state_dict keys to match the expected 'model.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("model."):
                new_state_dict[f"model.{k}"] = v
            else:
                new_state_dict[k] = v

        system = SpoofDetectorSystem(model=model_arch)
        system.load_state_dict(new_state_dict, strict=False)
        return system

    def eval(self) -> None:
        """Sets the model to evaluation mode (disables dropout and batch normalization)."""
        if self.model is not None:
            self.model.eval()

    def _chunk_audio(
        self, waveform: torch.Tensor, chunk_sec: int = 4, step_sec: int = 2, sr: int = 16000
    ) -> torch.Tensor:
        """
        Splits a long audio waveform into overlapping fixed-length segments.

        Args:
            waveform (torch.Tensor): Input audio signal of shape [1, T].
            chunk_sec (int): Duration of each segment in seconds.
            step_sec (int): Step size between segments in seconds (defines overlap).
            sr (int): Sampling rate of the audio.

        Returns:
            torch.Tensor: Batched segments of shape [N_chunks, 1, L].
        """
        chunk_len = chunk_sec * sr
        step_len = step_sec * sr

        # If audio is shorter than the required chunk length, pad with zeros
        if waveform.shape[1] <= chunk_len:
            pad_size = chunk_len - waveform.shape[1]
            return F.pad(waveform, (0, pad_size)).unsqueeze(0)

        chunks = []
        # Slide the window across the waveform
        for start in range(0, waveform.shape[1] - chunk_len + 1, step_len):
            chunk = waveform[:, start : start + chunk_len]
            chunks.append(chunk)

        # Ensure the final segment is included by taking the last 'chunk_len' samples
        if start + chunk_len < waveform.shape[1]:
            chunks.append(waveform[:, -chunk_len:])

        return torch.stack(chunks)

    @abstractmethod
    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for feature extraction (e.g., MFCC, LFCC, or Spectrograms).
        Must be implemented by subclasses.
        """
        pass

    def predict(
        self, features: Union[np.ndarray, torch.Tensor], threshold: float = 0.5
    ) -> Dict[str, Union[str, float]]:
        """
        Runs inference to determine if the audio is 'fake' or 'real'.

        Args:
            features (Union[np.ndarray, torch.Tensor]): Input waveform.
            threshold (float): Classification threshold. Values above this are considered fake.

        Returns:
            Dict: A dictionary containing the prediction label and confidence percentage.
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        # 1. Segment audio into chunks
        batched_features = self._chunk_audio(features)

        # 2. Extract features and move to target device
        batched_features = self.preprocess(batched_features).to(self.device)

        # 3. Model inference
        with torch.no_grad():
            logits = self.model(batched_features)
            probs = torch.sigmoid(logits) 

            # Aggregation: if any chunk is detected as fake, the whole file is suspicious
            final_prob = probs.max().item()

        # Debug logs
        print(f"DEBUG: Probabilities for all chunks: {probs.squeeze().tolist()}")
        print(f"Final Fake Probability = {final_prob}")

        prediction = "fake" if final_prob > threshold else "real"
        
        return {
            "prediction": prediction,
            "confidence": round(float(final_prob * 100), 2),
        }