from os import getenv

EXTRACTOR:str = getenv("EXTRACTOR","mel")
DETECTOR: str = getenv("DETECTOR","cnn")