import librosa
from pathlib import Path

def load_audio(path: Path):
    audio, sr = librosa.load(path,sr=16_000)
    return audio, sr