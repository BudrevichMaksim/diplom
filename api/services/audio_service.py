from pathlib import Path

from fastapi import UploadFile

from api.preprocessing.converter import convert_to_wav

TEMP_DIR: Path = Path("api/temp")

async def prepare_audio(file: UploadFile) -> Path:
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "audio.ogg"

    input_path = TEMP_DIR / filename

    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    wav_path = convert_to_wav(input_path)

    return wav_path
