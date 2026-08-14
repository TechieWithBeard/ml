from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
ML_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ML_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    model_provider: str = Field(
        default="ollama",
        alias="MODEL_PROVIDER",
    )

    chat_model: str = Field(
        default="gemma4:e4b",
        alias="OLLAMA_CHAT_MODEL",
    )

    embedding_model: str = Field(
        default="embeddinggemma:latest",
        alias="OLLAMA_EMBEDDING_MODEL",
    )

    ollama_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )

    huggingface_api_key: Optional[str] = Field(
        default=None,
        alias="HUGGINGFACEHUB_API_TOKEN",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings()
