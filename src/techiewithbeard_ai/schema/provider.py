from pydantic import BaseModel, SecretStr



class ModelConfig(BaseModel):
    provider: str = "ollama"  # "ollama", "hugging face", "openai", "openai_compatible"
    chat_model: str = "gemma4:e4b"
    embedding_model: str = "embeddinggemma:latest"
    ollama_url: str = "http://localhost:11434"
    hf_token: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    chroma_mode: str = "local"
    temperature: float = 0.1
    max_new_tokens: int = 1024

    def get(self, key: str, default=None):
        """
        Dict-style compatibility for older helpers that still call config.get().
        """
        return getattr(self, key, default)
