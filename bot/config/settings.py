from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration for the Telegram bot.
    
    Loads, validates, and type-casts environment variables from the system
    environment or a local '.env' file.
    """

    # The authorization token provided by Telegram's BotFather. Required.
    bot_token: str = Field(alias="BOT_TOKEN")

    # The base URL pointing to your FastAPI inference service.
    api_path: str = Field(
        default="http://127.0.0.1:8000",
        alias="API_PATH"
    )

    # Maximum allowed duration (in seconds) for incoming voice messages.
    max_voice_duration: int = Field(
        default=10,
        alias="MAX_VOICE_DURATION"
    )

    # Minimum allowed duration (in seconds) for incoming voice messages.
    min_voice_duration: int = Field(
        default=2,
        alias="MIN_VOICE_DURATION"
    )

    # Pydantic-specific configuration settings
    model_config = SettingsConfigDict(
        env_file=".env",            # Read variables from a local .env file automatically
        extra="ignore"              # Safely ignore extra environment variables not defined here
    )
