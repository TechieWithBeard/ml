import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from techiewithbeard_ai.agents.agents import get_embedding_model
from techiewithbeard_ai.schema.provider import ModelConfig


load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[3] / ".env"
)


def get_collection_name(config: ModelConfig) -> str:
    """
    Create a unique collection for each embedding provider/model.

    This prevents embedding dimension mismatches when switching
    between embedding models.
    """

    provider = (
        config.provider
        .lower()
        .replace(" ", "")
        .replace("-", "")
    )

    model = (
        config.embedding_model
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
    )

    return f"resume_{provider}_{model}"


def get_cloud_chroma_connection_kwargs() -> dict:
    """
    Return Chroma Cloud connection configuration.
    """

    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if not api_key:
        raise ValueError("CHROMA_API_KEY is not configured.")

    if not tenant:
        raise ValueError("CHROMA_TENANT is not configured.")

    if not database:
        raise ValueError("CHROMA_DATABASE is not configured.")

    return {
        "chroma_cloud_api_key": api_key,
        "tenant": tenant,
        "database": database,
    }


def get_vector_store(
    config: ModelConfig,
) -> Chroma:

    embeddings = get_embedding_model(config)

    collection_name = get_collection_name(config)

    chroma_mode = config.chroma_mode

    print(f"Provider: {config.provider}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Chroma mode: {chroma_mode}")
    print(f"Chroma collection: {collection_name}")

    # --------------------------------
    # Local Chroma
    # --------------------------------

    if chroma_mode == "local":

        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./data/chroma",
        )

    # --------------------------------
    # Chroma Cloud
    # --------------------------------

    if chroma_mode == "cloud":

        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            **get_cloud_chroma_connection_kwargs(),
        )

    raise ValueError(
        f"Unsupported CHROMA_MODE: {chroma_mode}. "
        f"Expected 'local' or 'cloud'."
    )


def inspect_embeddings(
    config: ModelConfig,
    chunks: list[str],
) -> Embeddings:

    embeddings = get_embedding_model(config)

    if not chunks:
        print("No chunks available.")
        return embeddings

    embeds = embeddings.embed_documents(chunks)

    print(
        f"Number of embeddings returned: {len(embeds)}"
    )

    print(
        f"Embedding dimension: {len(embeds[0])}"
    )

    return embeddings


def upload_embeddings_to_chroma(
    config: ModelConfig,
    texts: list[str],
    metadatas: list[dict],
    ids: Optional[list[str]] = None,
):
    """
    Embed and upload text to the configured Chroma
    backend.
    """

    vector_store = get_vector_store(config)

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

    collection_name = get_collection_name(config)

    print(
        f"Uploaded {len(texts)} documents to "
        f"Chroma collection '{collection_name}'."
    )

    return vector_store