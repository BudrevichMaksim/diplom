from pathlib import Path
from typing import List, Tuple, Union


def collect_labeled_files(
    root_dir: Union[str, Path], extensions: List[str]
) -> List[Tuple[Path, int]]:
    """
    Scans class subdirectories ('fake' and 'real') and aggregates labeled file paths.

    Expects the root directory to contain 'fake' and/or 'real' subfolders.
    Files matching the provided extensions are gathered and paired with an integer 
    label: 1 for fake (spoofed) and 0 for real (bona fide).

    Args:
        root_dir (Union[str, Path]): Path to the dataset partition root directory.
        extensions (List[str]): Glob patterns or extensions to search for (e.g., ["*.wav"]).

    Returns:
        List[Tuple[Path, int]]: A list of tuples containing the absolute Path object 
            and its corresponding class integer label (0 or 1).
    """
    root = Path(root_dir)
    samples: List[Tuple[Path, int]] = []

    # Map target binary targets: 1 for Fake/Spoofed, 0 for Real/Bona fide
    class_mapping = [("fake", 1), ("real", 0)]

    for label_name, label_value in class_mapping:
        class_dir = root / label_name

        if not class_dir.exists():
            continue

        for ext in extensions:
            for path in class_dir.glob(ext):
                # Ensure we only append actual files, skipping nested directories if any
                if path.is_file():
                    samples.append((path, label_value))

    return samples