import subprocess
from pathlib import Path

def convert_to_wav(input_path: Path) -> Path:
    """
    Converts an input audio file to a standardized WAV format using FFmpeg.

    The output file will have a .wav extension, a 16,000 Hz sampling rate, 
    and a single audio channel (mono).

    Args:
        input_path (Path): Path to the source audio file (mp3, m4a, flac, etc.).

    Returns:
        Path: The path to the newly created .wav file.

    Raises:
        subprocess.CalledProcessError: If the FFmpeg command fails.
        FileNotFoundError: If FFmpeg is not installed on the system.
    """
    # Define the output path by changing the extension to .wav
    output_path = input_path.with_suffix(".wav")

    # ffmpeg command arguments:
    # -y: Overwrite output files without asking
    # -i: Input file path
    # -ar: Set audio sampling rate to 16000 Hz
    # -ac: Set number of audio channels to 1 (mono)
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        str(output_path)
    ], check=True)  # check=True raises an error if the command fails

    return output_path