from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Any
from torch.utils.data import Dataset


class BaseAudioDataset(Dataset, ABC):
    """
    Abstract base class for audio dataset implementations.

    Handles unified initialization, structures the directory discovery loop,
    and enforces baseline target verification patterns across custom dataset subclasses.
    """

    def __init__(self, root_dir: Union[str, Path]):
        """
        Initializes the base audio dataset.

        Args:
            root_dir (Union[str, Path]): Path to the dataset root directory.

        Raises:
            RuntimeError: If zero valid samples are found by the subclass collector.
        """
        self.root_dir = root_dir
        self.samples = self._collect_samples()

        if not self.samples:
            raise RuntimeError(f"No samples found in: {root_dir}")

    @abstractmethod
    def _collect_samples(self) -> List[Any]:
        """
        Discovers, filters, and registers valid dataset sample targets.

        Returns:
            List[Any]: Structural collection of verified tracking targets.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Retrieves, transforms, and packages a single sample at the given index.

        Args:
            idx (int): Structural element lookup indicator query index.

        Returns:
            Any: Formatted sample block, typically a tensor mapping dictionary.
        """
        pass

    def __len__(self) -> int:
        """Returns the total number of collected data samples."""
        return len(self.samples)
