from pathlib import Path
from typing import Dict, List, Tuple, Union
import warnings
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """
    Scans a dataset root directory and performs stratified partitioning
    of files into training, validation, and testing splits.
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        extensions: Tuple[str, ...] = ("*.wav", "*.flac", "*.mp3", "*.ogg"),
        random_state: int = 38,
    ):
        """
        Initializes the dataset splitter.

        Args:
            input_dir (Union[str, Path]): Path to the root directory containing
                'real' and 'fake' subfolders.
            extensions (Tuple[str, ...]): Glob patterns for target audio formats.
            random_state (int): Seed used by the random number generator for reproducibility.
        """
        self.input_dir = Path(input_dir)
        self.extensions = extensions
        self.random_state = random_state

    def _gather_files(self) -> Tuple[List[Path], List[str]]:
        """
        Gathers all valid file paths and their associated class target names
        from 'real' and 'fake' subdirectories.

        Returns:
            Tuple[List[Path], List[str]]: A tuple containing a list of target file
                Paths and a matching list of structural label strings.

        Raises:
            ValueError: If no files matching the specified extensions are found.
        """
        dataset_files: List[Path] = []
        labels: List[str] = []

        for label_name in ["real", "fake"]:
            class_dir = self.input_dir / label_name
            if not class_dir.exists():
                warnings.warn(f"Class directory {class_dir} was not located.")
                continue

            for ext in self.extensions:
                for file_path in class_dir.glob(ext):
                    if file_path.is_file():
                        dataset_files.append(file_path)
                        labels.append(label_name)

        if not dataset_files:
            raise ValueError(
                f"No audio files discovered matching the specified extensions in {self.input_dir}"
            )

        return dataset_files, labels

    def get_splits(
        self, val_size: float = 0.1, test_size: float = 0.1
    ) -> Dict[str, List[Tuple[Path, str]]]:
        """
        Partitions the collected dataset assets into training, validation, and testing subsets.

        Args:
            val_size (float): Proportion of the total dataset to allocate for validation.
            test_size (float): Proportion of the total dataset to allocate for testing.

        Returns:
            Dict[str, List[Tuple[Path, str]]]: A mapping of split keys ('training',
                'validation', 'testing') to collections of paired file paths and labels.
        """
        files, labels = self._gather_files()

        # 1. Isolate the testing partition using stratified target distribution
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            files,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=self.random_state,
        )

        # 2. Recalibrate partition sizes and isolate the validation set from remaining samples
        val_ratio_adjusted = val_size / (1.0 - test_size)
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files,
            train_val_labels,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels,
            random_state=self.random_state,
        )

        # 3. Construct and map final structural collections
        return {
            "training": list(zip(train_files, train_labels)),
            "validation": list(zip(val_files, val_labels)),
            "testing": list(zip(test_files, test_labels)),
        }
