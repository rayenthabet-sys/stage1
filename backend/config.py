from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # These will be loaded from .env or environment variables
    OPENED_CLIENT_ID: str = ""
    OPENED_CLIENT_SECRET: str = ""

    # Load from .env file in the project root
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
