from pathlib import Path
from fastapi import UploadFile
from api.preprocessing.converter import convert_to_wav

# Global directory for temporary audio storage
TEMP_DIR: Path = Path("api/temp")

async def prepare_audio(file: UploadFile) -> Path:
    """
    Handles the ingestion and standardization of an uploaded audio file.

    This process includes:
    1. Creating a temporary storage directory if it doesn't exist.
    2. Writing the raw uploaded bytes to a local file.
    3. Converting the raw file into a standardized WAV format (16kHz, Mono).

    Args:
        file (UploadFile): The raw file stream received from the client.

    Returns:
        Path: The filesystem path to the processed .wav file ready for inference.
    """
    # Ensure the workspace directory exists
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Use a default filename if the upload doesn't provide one
    filename = file.filename or "audio.ogg"
    input_path = TEMP_DIR / filename

    # Asynchronously read the uploaded bytes and save them to disk
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # Standardize the file format using the FFmpeg-based converter
    wav_path = convert_to_wav(input_path)

    return wav_path