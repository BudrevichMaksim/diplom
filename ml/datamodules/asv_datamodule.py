import lightning as L
import torch
from torch.utils.data import DataLoader
from pathlib import Path


class ASV5DataModule(L.LightningDataModule):
    """
    DataModule tailored for the ASVspoof5 challenge dataset structure.

    Automates path mapping for explicit challenge protocols (train, dev, eval)
    and wraps them into optimized DataLoader pipelines.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_class,
        batch_size: int = 32,
        num_workers: int = 4,
        dataset_kwargs: dict = None,
        collate_fn=None,
    ):
        """
        Initializes the ASV5DataModule paths and settings.

        Args:
            root_dir (str): Base path to the ASVspoof5 dataset root directory.
            dataset_class (Type[Dataset]): Dataset class to initialize for each phase.
            batch_size (int): Image/audio sample batch size per loader.
            num_workers (int): Parallel CPU worker count for data fetching.
            dataset_kwargs (dict, optional): Keyword configurations passed directly to the dataset.
            collate_fn (callable, optional): Custom collate function to process batched tensors.
        """
        super().__init__()

        self.root_dir = Path(root_dir)
        self.dataset_class = dataset_class

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs or {}
        self.collate_fn = collate_fn

        # Predefined file names and structures according to ASVspoof5 challenge guidelines
        self.train_protocol = self.root_dir / "ASVspoof5.train.tsv"
        self.train_dir = self.root_dir / "flac_T"

        self.val_protocol = self.root_dir / "ASVspoof5.dev.track_1.tsv"
        self.val_dir = self.root_dir / "flac_D"

        self.test_protocol = self.root_dir / "ASVspoof5.eval.track_1.tsv"
        self.test_dir = self.root_dir / "flac_E_eval"

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """
        Builds the corresponding dataset targets using targeted TSV metadata tracking.

        Args:
            stage (str, optional): Target execution tracking state ('fit', 'test').
        """
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_class(
                protocol_path=self.train_protocol,
                flac_dir=self.train_dir,
                is_train=True,
                **self.dataset_kwargs,
            )

            self.val_dataset = self.dataset_class(
                protocol_path=self.val_protocol,
                flac_dir=self.val_dir,
                is_train=False,
                **self.dataset_kwargs,
            )

        if stage == "test":
            self.test_dataset = self.dataset_class(
                protocol_path=self.test_protocol,
                flac_dir=self.test_dir,
                is_train=False,
                **self.dataset_kwargs,
            )

    def train_dataloader(self) -> DataLoader:
        return self._create_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._create_loader(self.test_dataset, shuffle=False)

    def _create_loader(self, dataset, shuffle: bool = False) -> DataLoader:
        """
        Helper method to isolate boilerplate parameters for DataLoader creation.

        Args:
            dataset (Dataset): Target underlying data container.
            shuffle (bool): Toggles randomized batch sampling and training drop_last rule.

        Returns:
            DataLoader: Synchronized and padded pipeline data loader instance.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            # drop_last protects training state stability against small residual tail batches
            drop_last=shuffle,
        )
