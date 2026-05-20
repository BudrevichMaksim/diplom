from pathlib import Path
from typing import List, Tuple, Union


class ASV5Splitter:
    """
    Parses ASVspoof 5 protocol metadata files and aligns them with local audio samples.

    Filters trial metadata records to verify that corresponding audio assets actually
    exist on disk, mapping target string tags ('spoof' / 'bonafide') into clean
    machine-learning-ready binary indicators.
    """

    def __init__(self, protocol_path: Union[str, Path], flac_dir: Union[str, Path]):
        """
        Initializes the ASVspoof 5 protocol splitter utility.

        Args:
            protocol_path (Union[str, Path]): Path to the space-separated metadata TXT file.
            flac_dir (Union[str, Path]): Root path pointing to the downloaded .flac files.
        """
        self.protocol_path = Path(protocol_path)
        self.flac_dir = Path(flac_dir)

    def get_available_samples(self) -> List[Tuple[Path, int]]:
        """
        Parses the protocol trial file, filtering for locally available FLAC samples.

        Expects standard ASVspoof space-delimited text structure where:
        - index 1: File name stem key identifier (e.g., 'E_5000001')
        - index 8: Evaluation system target state string ('spoof' or 'bonafide')

        Returns:
            List[Tuple[Path, int]]: A list of absolute file Path targets coupled
                with their classification label integer (1 for spoof, 0 for bonafide).

        Raises:
            FileNotFoundError: If the designated metadata protocol path does not exist.
        """
        samples: List[Tuple[Path, int]] = []

        if not self.protocol_path.exists():
            raise FileNotFoundError(
                f"Protocol file not located at: {self.protocol_path}"
            )

        with open(self.protocol_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                # Dynamically bypass file header rows or malformed telemetry sequences
                if len(parts) < 9:
                    continue

                file_name = parts[1]  # Structural identity token string
                label_str = parts[8]  # Primary indicator criteria assignment string

                # Align against local download state boundaries
                file_path = self.flac_dir / f"{file_name}.flac"
                if file_path.is_file():
                    label = 1 if label_str == "spoof" else 0
                    samples.append((file_path, label))

        return samples
