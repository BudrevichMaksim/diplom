import lightning as L
import torch
from torch.utils.data import DataLoader, random_split


class AudioDataModule(L.LightningDataModule):
    """
    A unified LightningDataModule for audio processing pipelines.

    Supports two modes of operation:
    1. Loading data from explicit, pre-split directories (train, val, test).
    2. Loading a single monolithic dataset from a root directory and splitting
       it randomly based on predefined fractional ratios.
    """

    def __init__(
        self,
        dataset_class,
        train_dir: str = None,
        val_dir: str = None,
        test_dir: str = None,
        root_dir: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        dataset_kwargs: dict = None,
        collate_fn=None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        """
        Initializes the AudioDataModule.

        Args:
            dataset_class (Type[Dataset]): The PyTorch Dataset class subclass to instantiate.
            train_dir (str, optional): Path to the explicit training data directory.
            val_dir (str, optional): Path to the explicit validation data directory.
            test_dir (str, optional): Path to the explicit test data directory.
            root_dir (str, optional): Path to the shared root directory for random splitting.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of CPU subprocesses for data loading.
            dataset_kwargs (dict, optional): Extra configuration parameters passed to the dataset.
            collate_fn (callable, optional): Custom function to merge a list of samples into a batch.
            train_split (float): Fraction of data allocated to training (used if root_dir is set).
            val_split (float): Fraction of data allocated to validation (used if root_dir is set).
            pin_memory (bool): If True, copies tensors into device pinned memory before returning.
            persistent_workers (bool): If True, maintains data loader worker processes across epochs.
        """
        super().__init__()
        self.dataset_class = dataset_class
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs or {}
        self.collate_fn = collate_fn
        self.train_split = train_split
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """
        Instantiates datasets and builds splits depending on the data pipeline mode.

        Args:
            stage (str, optional): Lightning execution stage ('fit', 'validate', 'test').
        """
        if self.train_dir is not None:
            if stage == "fit" or stage is None:
                self.train_dataset = self.dataset_class(
                    self.train_dir, **self.dataset_kwargs
                )
                self.val_dataset = self.dataset_class(
                    self.val_dir, **self.dataset_kwargs
                )

            if stage == "test" or stage is None:
                self.test_dataset = self.dataset_class(
                    self.test_dir, **self.dataset_kwargs
                )
        else:
            full_dataset = self.dataset_class(self.root_dir, **self.dataset_kwargs)
            total = len(full_dataset)

            train_len = int(total * self.train_split)
            val_len = int(total * self.val_split)
            test_len = total - train_len - val_len

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataset data loader configuration."""
        return self._create_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataset data loader configuration."""
        return self._create_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataset data loader configuration."""
        return self._create_loader(self.test_dataset, shuffle=False)

    def _create_loader(self, dataset, shuffle: bool = False) -> DataLoader:
        """
        Helper method to generate standard DataLoader objects uniformly.

        Args:
            dataset (Dataset): The underlying dataset to wrap.
            shuffle (bool): Whether to randomize sample orders every epoch.

        Returns:
            DataLoader: Configured PyTorch DataLoader instance.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            # Guard against PyTorch runtime exceptions when persistent_workers=True but num_workers=0
            persistent_workers=(self.num_workers > 0 and self.persistent_workers),
        )
