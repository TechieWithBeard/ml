from dataclasses import dataclass

from pydantic import BaseModel, SecretStr



class ModelConfig(BaseModel):
    provider: str = "ollama"
    chat_model: str = "gemma4:e4b"
    embedding_model: str = "embeddinggemma:latest"
    ollama_url: str = "http://localhost:11434"
    hf_token: SecretStr | None = None
    chroma_mode: str = "local"