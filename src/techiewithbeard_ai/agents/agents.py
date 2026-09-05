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
    max_tokens = config.max_new_tokens or 4096
    
    print(
        f"Initializing {provider} model: {config.chat_model}, "
        f"max_new_tokens={max_tokens}"
    )
    
    if provider == "ollama":
        return ChatOllama(
            model=config.chat_model,
            base_url=config.ollama_url,
            temperature=config.temperature,
            num_predict=max_tokens,
            num_ctx=16384,
            keep_alive="30m"
        )
    
    if provider == "hugging face":
        if not config.hf_token:
            raise ValueError(
                "Hugging Face token is required."
            )
        llm = HuggingFaceEndpoint(
            model=config.chat_model,
            huggingfacehub_api_token=config.hf_token.get_secret_value(),
            temperature=config.temperature,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

        return ChatHuggingFace(
            llm=llm,
        )
    
    if provider in ["openai", "openai_compatible", "openapi"]:
        api_key = config.openai_api_key.get_secret_value() if config.openai_api_key else os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Please set it in configuration or OPENAI_API_KEY environment variable.")
        
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.chat_model or "gpt-4o-mini",
            api_key=api_key,
            base_url=config.openai_base_url or None,
            temperature=config.temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Unsupported chat provider: {config.provider}"
    )