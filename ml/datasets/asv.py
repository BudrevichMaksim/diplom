import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple, Dict, Union
from ml.datasets.augs import RawAudioAugmentor
from ml.datasets.base import BaseAudioDataset


class ASVSpoof5Dataset(BaseAudioDataset):
    """
    Dataset class for processing and loading ASVspoof5 challenge audio tracks.

    Parses challenge-specific TSV metadata protocols, matches tracking records
    with corresponding FLAC audio files, handles mono-downmixing, adaptive
    resampling, uniform padding/truncation, and training augmentations.
    """

    def __init__(
        self,
        protocol_path: Union[str, Path],
        flac_dir: Union[str, Path],
        target_sr: int = 16000,
        max_seconds: int = 4,
        is_train: bool = False,
    ):
        """
        Initializes the ASVSpoof5 dataset instance.

        Args:
            protocol_path (Union[str, Path]): Path to the challenge metadata TSV file.
            flac_dir (Union[str, Path]): Directory containing the raw .flac audio files.
            target_sr (int): Target sampling rate to conform all audio signals to.
            max_seconds (int): Maximum allowed audio sequence duration in seconds.
            is_train (bool): If True, enables raw waveform online data augmentations.

        Raises:
            RuntimeError: If zero valid audio samples matching the protocol are discovered.
        """
        self.flac_dir = Path(flac_dir)
        self.target_sr = target_sr
        self.max_samples = target_sr * max_seconds
        self.is_train = is_train
        self.protocol_path = Path(protocol_path)

        self.augmentor = RawAudioAugmentor() if is_train else None

        self.num_real = 0
        self.num_fake = 0

        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found. Check protocol={self.protocol_path} and flac_dir={self.flac_dir}"
            )

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """
        Parses the protocol file and builds absolute validation paths for tracking files.

        Returns:
            List[Tuple[Path, int]]: A collection of pairs containing verified file paths
                and mapped binary targets (0 for bonafide, 1 for spoof).
        """
        samples = []

        with open(self.protocol_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 9:
                    continue

                file_id = parts[1]
                label_str = parts[8]

                file_path = self.flac_dir / f"{file_id}.flac"
                if not file_path.exists():
                    continue

                if label_str == "bonafide":
                    label = 0
                    self.num_real += 1
                elif label_str == "spoof":
                    label = 1
                    self.num_fake += 1
                else:
                    continue

                samples.append((file_path, label))

        return samples

    @property
    def pos_weight(self) -> float:
        """
        Computes the ratio balancing factor for classes.

        Returns:
            float: Imbalance multiplier scaling weight (Real count / Fake count).
        """
        if self.num_fake == 0:
            return 1.0
        return self.num_real / self.num_fake

    def __len__(self) -> int:
        """Returns the total number of verified audio samples within the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads, normalizes, and wraps a target raw audio sample sequence.

        Args:
            idx (int): Structural sample retrieval indicator query index.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing normalized 1D 'features'
                waveform tensors and scalar 'label' indicator tensors.
        """
        path, label = self.samples[idx]

        waveform, sr = torchaudio.load(path)

        # Enforce sampling rate consistency across mixed records
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)

        # Force conversion to single-channel mono signal representations
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Uniform sequence length standardization via cropping or trailing zero padding
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, : self.max_samples]
        else:
            pad = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        if self.is_train and self.augmentor:
            waveform = self.augmentor(waveform)

        return {
            "features": waveform.squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }
