from pathlib import Path
from typing import Tuple, Union
import torch
from torch import nn
import torchaudio
from tqdm.auto import tqdm

from ml.feature_extractors.splitter import DatasetSplitter


class FeaturePipeline:
    """
    End-to-end processing pipeline for audio datasets to extract and save features.
    """

    def __init__(
        self,
        extractor: nn.Module,
        output_dir: Union[str, Path],
        target_sr: int = 16000,
    ):
        """
        Initializes the feature extraction pipeline.

        Args:
            extractor (nn.Module): Feature extraction module (e.g., MelSpectrogramExtractor).
            output_dir (Union[str, Path]): Root directory where extracted tensors will be saved.
            target_sr (int): Target sampling rate for the audio files.
        """
        super().__init__()
        self.extractor = extractor
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr

    def _load_and_preprocess_audio(self, file_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Loads, resamples, and downmixes an audio file to mono.

        Args:
            file_path (Path): Path to the source audio file.

        Returns:
            Tuple[torch.Tensor, int]: Preprocessed waveform tensor and its sampling rate.
        """
        waveform, sr = torchaudio.load(file_path)

        if sr != self.target_sr:
            # Use functional resample to avoid nn.Module allocation overhead inside loops
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )
            sr = self.target_sr

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sr

    def process_dataset(
        self,
        splitter: DatasetSplitter,
        val_size: float = 0.1,
        test_size: float = 0.1,
    ) -> None:
        """
        Runs the extraction and partition saving process for the entire dataset.

        Args:
            splitter (DatasetSplitter): Dataset splitter instance to fetch data splits.
            val_size (float): Proportion of the validation split.
            test_size (float): Proportion of the test split.
        """
        splits = splitter.get_splits(val_size=val_size, test_size=test_size)

        for split_name, files_list in splits.items():
            print(f"\nProcessing split: {split_name} ({len(files_list)} files)")

            for file_path, label in tqdm(files_list, desc=split_name):
                try:
                    # 1. Load and normalize audio signal
                    waveform, _ = self._load_and_preprocess_audio(file_path)

                    # 2. Extract features without tracking gradients
                    with torch.no_grad():
                        features = self.extractor(waveform)

                    # 3. Define directories and save the tensor
                    # Convert integer label to string to prevent Path concatenation TypeError
                    save_dir = self.output_dir / split_name / str(label)
                    save_dir.mkdir(parents=True, exist_ok=True)

                    save_path = save_dir / f"{file_path.stem}.pt"
                    torch.save(features, save_path)

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        print(f"\nProcessing completed. Tensors saved to: {self.output_dir.absolute()}")
