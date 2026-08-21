import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_ollama import ChatOllama, OllamaEmbeddings

from techiewithbeard_ai.schema.provider import ModelConfig


def get_embedding_model(config: ModelConfig):
    provider = config.provider.lower()
    if provider == "ollama":
        return OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_url,
        )
        
    if provider == "hugging face":

        return HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )
    raise ValueError(
        f"Unsupported embedding provider: {config.provider}"
    )


def get_chat_model(config: ModelConfig):
    provider = config.provider.lower()
    print(
    f"HF model={config.chat_model}, "
    f"max_new_tokens={config.max_new_tokens}"
)
    if provider == "ollama":
        return ChatOllama(
            model=config.chat_model,
            base_url=config.ollama_url,
            temperature=config.temperature,
            num_predict=config.max_new_tokens,
            num_ctx=16384
        )
    if provider == "hugging face":
        if not config.hf_token:
            raise ValueError(
                "Hugging Face token is required."
            )
        llm = HuggingFaceEndpoint(
                model=config.chat_model,
                huggingfacehub_api_token=config.hf_token.get_secret_value(),
                    temperature=0.0,
                    max_new_tokens=1024,
                )

        return ChatHuggingFace(
                    llm=llm,
                )
    raise ValueError(
        f"Unsupported chat provider: {config.provider}"
    )