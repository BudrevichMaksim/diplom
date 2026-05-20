from pathlib import Path
from typing import Dict, List, Tuple, Union
import torch
import torch.nn.functional as F
import torchaudio
from ml.datasets.base import BaseAudioDataset
from ml.datasets.utils import collect_labeled_files


class RawAudioDataset(BaseAudioDataset):
    """
    Generic dataset for loading and conforming raw multi-format audio files.

    Scans for various consumer and compressed audio extensions, handles dynamic
    resampling to a unified target rate, forces mono downmixing, and ensures
    uniform temporal dimensions through trimming or zero-padding.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        target_sr: int = 16000,
        max_seconds: int = 4,
    ):
        """
        Initializes the raw audio dataset locator.

        Args:
            root_dir (Union[str, Path]): Path to the directory holding the raw audio folders.
            target_sr (int): Target sampling rate to conform all audio signals to.
            max_seconds (int): Maximum allowed audio sequence duration in seconds.
        """
        self.target_sr = target_sr
        self.max_samples = target_sr * max_seconds

        # Super class init triggers self._collect_samples()
        super().__init__(root_dir)

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """
        Scans the root directory for supported lossy and lossless raw audio targets.

        Returns:
            List[Tuple[Path, int]]: A collection of paired absolute paths and
                their respective target integer labels.
        """
        return collect_labeled_files(
            self.root_dir,
            ["*.wav", "*.mp3", "*.ogg", "*.m4a"],
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads, normalizes, and packages a single raw audio file.

        Args:
            idx (int): Structural sample lookup indicator query index.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "features": Squeezed 1D raw waveform tensor of shape [Time].
                - "label": A scalar float32 indicator tensor representing the target target.

        Raises:
            RuntimeError: If torchaudio fails to correctly read the track file.
        """
        path, label = self.samples[idx]

        waveform, sr = torchaudio.load(path)

        # Enforce sampling rate consistency across mixed-source files
        if sr != self.target_sr:
            # Using functional resample avoids instantiating an nn.Module object inside the fetch loop
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )

        # Force conversion to single-channel mono signal representations
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = self._pad_or_trim(waveform)

        return {
            "features": waveform.squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Standardizes the length of the waveform tensor to a uniform sample window size.

        Args:
            waveform (torch.Tensor): Audio tensor of shape [Channels, Time].

        Returns:
            torch.Tensor: Conformed length audio tensor of shape [Channels, Max_Samples].
        """
        current_length = waveform.shape[1]

        if current_length > self.max_samples:
            return waveform[:, : self.max_samples]

        pad_size = self.max_samples - current_length
        return F.pad(waveform, (0, pad_size))
