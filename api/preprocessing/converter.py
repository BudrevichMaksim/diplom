import subprocess
from pathlib import Path

def convert_to_wav(input_path: Path) -> Path:
    output_path = input_path.with_suffix(".wav")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path)
    ])

    return output_path