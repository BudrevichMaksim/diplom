from pathlib import Path
from typing import Dict, List, Tuple, Union
import torch
from ml.datasets.base import BaseAudioDataset
from ml.datasets.utils import collect_labeled_files


class PreprocessedDataset(BaseAudioDataset):
    """
    Dataset for loading pre-extracted audio features saved as PyTorch tensors (.pt).

    Inherits from BaseAudioDataset to scan, register, and securely load tensor
    files on-the-fly. Automatically ensures that all 2D spectrograms or embeddings
    are expanded to include a consistent channel dimension.
    """

    def __init__(self, root_dir: Union[str, Path]):
        """
        Initializes the preprocessed feature dataset.

        Args:
            root_dir (Union[str, Path]): Path to the directory holding the preprocessed .pt files.
        """
        super().__init__(root_dir)

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """
        Scans the root directory for preprocessed PyTorch files and pairs them with labels.

        Returns:
            List[Tuple[Path, int]]: A list of pairs containing verified file paths
                and their corresponding target integer labels.
        """
        return collect_labeled_files(
            self.root_dir,
            ["*.pt"],
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads, normalizes, and formats a single pre-extracted feature tensor block.

        Args:
            idx (int): Structural sample lookup indicator query index.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "features": Float32 feature tensor guaranteed to be 3D [Channels, Freq/Dim, Time].
                - "label": A scalar float32 indicator tensor representing the target target.
        """
        path, label = self.samples[idx]

        # Securely deserialize the tensor file to prevent arbitrary code execution risks
        features = torch.load(
            path,
            weights_only=True,
        )

        # Ensure a channel dimension exists (e.g., converting [Freq, Time] -> [1, Freq, Time])
        # to ensure downstream 2D convolutional layers receive standard structural footprints.
        if features.ndim == 2:
            features = features.unsqueeze(0)

        return {
            "features": features.float(),
            "label": torch.tensor(
                label,
                dtype=torch.float32,
            ),
        }
