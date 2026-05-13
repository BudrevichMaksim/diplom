from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """
    Application settings and environment configuration.
    Defined using Pydantic for automatic validation and type hinting.
    """

    # We can restrict the values to only what our API supports
    EXTRACTOR: Literal["mel"] = "mel"
    DETECTOR: str = "rcnn"

    # Configuration for the Settings class itself
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Create a singleton instance to be used across the app
settings = Settings()
