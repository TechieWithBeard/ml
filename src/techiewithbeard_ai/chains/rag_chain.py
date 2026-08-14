import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings
from  pydantic import BaseModel, Field
from techiewithbeard_ai.agents.agents import get_chat_model, get_embedding_model
from techiewithbeard_ai.schema.provider import ModelConfig

from techiewithbeard_ai.retrievers.embeddings import get_vector_store

load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")

class ResumeAnalyser(BaseModel):
    candidate_name: str | None = Field(
        default=None,
        description="Candidate's full name."
    )

    skills: list[str] = Field(
        default_factory=list,
        description="Technical skills found in the resume."
    )

    experience: list[str] = Field(
        default_factory=list,
        description="Relevant work experience."
    )

    links: list[str] = Field(
        default_factory=list,
        description="LinkedIn, GitHub, portfolio and other links."
    )

    query: str = Field(
        description="The user's question."
    )

    observation: str = Field(
        description="Answer to the user's question."
    )
    
# def get_vector_store(provider_config:ModelConfig,collection_name: str = "example_collection") -> Chroma:
#     """Return the configured cloud-backed Chroma vector store for the ML app."""
#     embeddings = get_embedding_model(provider_config)
#     print(f"Using embeddings: {embeddings} from a common path")
#     api_key = os.getenv("CHROMA_API_KEY")
#     tenant = os.getenv("CHROMA_TENANT")
#     database = os.getenv("CHROMA_DATABASE")
    
#     return Chroma(
#         collection_name=collection_name,
#         embedding_function=embeddings,
#         chroma_cloud_api_key=api_key,
#         tenant=tenant,
#         database=database,
#     )


def _collection_count(vector_store: Any) -> int:
    """Return the number of indexed records using the Chroma API supported in this runtime."""
    if hasattr(vector_store, "count"):
        return int(vector_store.count())

    collection = getattr(vector_store, "_collection", None)
    if collection is not None and hasattr(collection, "count"):
        return int(collection.count())

    try:
        return len(vector_store.get()["documents"])
    except Exception:
        return 0


def _get_relevant_docs(provider_config:ModelConfig,question: str, k: int = 5) -> list[tuple[Any, float]]:
    vector_store = get_vector_store(provider_config)
    if _collection_count(vector_store) == 0:
        return []
    return vector_store.similarity_search_with_score(question, k=k)


def _build_rag_answer(
    provider_config: ModelConfig,
    question: str,
    docs: list[tuple[Any, float]],
) -> str:

    if not docs:
        return (
            "No matching data found in the vector store. "
            "Please index the PDF content first."
        )

    context = "\n\n".join(
        f"Document {index}:\n"
        f"{doc.page_content}\n"
        f"Metadata: {doc.metadata}"
        for index, (doc, _) in enumerate(docs, start=1)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are answering from the retrieved document context.

                Be concise and grounded in the provided context.

                If the answer is not present in the context,
                say:
                "I couldn't find it in the retrieved context."
                """,
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}",
            ),
        ]
    )

    llm = get_chat_model(provider_config)

    chain = prompt | llm | StrOutputParser()

    try:
        return chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

    except Exception as ex:
        print(f"LLM invocation failed: {ex}")

        return (
            f"Unable to generate an answer. "
            f"Error: {ex}"
        )


def _build_rag_answer_with_pydantic(
    provider_config: ModelConfig,
    question: str,
    docs: list[tuple[Any, float]],
) -> ResumeAnalyser:

    if not docs:
        return ResumeAnalyser(
            query=question,
            observation=(
                "No matching data found in the vector store. "
                "Please index the PDF content first."
            ),
        )

    context = "\n\n".join(
        f"Document {index}:\n"
        f"{doc.page_content}\n"
        f"Metadata: {doc.metadata}"
        for index, (doc, _) in enumerate(docs, start=1)
    )

    parser = PydanticOutputParser(
        pydantic_object=ResumeAnalyser
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a resume analysis assistant.

Answer the user's question using ONLY the supplied resume context.

Extract information when available.

Important:
- candidate_name: candidate's full name.
- skills: technical skills found in the resume.
- experience: relevant experience, companies, roles, or positions.
- links: LinkedIn, GitHub, portfolio or other links.
- query: copy the user's question exactly.
- observation: directly answer the user's question.

Do not invent information.

If a field cannot be found:
- candidate_name -> null
- skills -> []
- experience -> []
- links -> []

Return ONLY valid JSON.

{format_instructions}
""",
            ),
            (
                "human",
                """
Question:
{question}

Resume Context:
{context}
""",
            ),
        ]
    )

    llm = get_chat_model(provider_config)

    chain = prompt | llm | parser

    try:
        result = chain.invoke(
            {
                "question": question,
                "context": context,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        if isinstance(result, ResumeAnalyser):
            return result

        return ResumeAnalyser.model_validate(result)

    except Exception as ex:

        print(f"Structured output error: {ex}")

        return ResumeAnalyser(
            query=question,
            observation=f"Unable to generate an answer: {ex}",
        )


def build_rag_chain(provider_config:ModelConfig):
    """Return a reusable chain that checks the vector store and answers from retrieved context."""
    
    def retrieve_documents(payload: dict[str, Any]) -> dict[str, Any]:
        question = payload["query"]
        docs = _get_relevant_docs(provider_config,question)
        return {
            "query": question,
            "retrieved_documents": docs,
        }

    def finalize_answer(payload: dict[str, Any]) -> dict[str, Any]:
        question = payload["query"]
        docs = payload["retrieved_documents"]
        # answer = _build_rag_answer(question, docs)
        answer = _build_rag_answer_with_pydantic(provider_config,question, docs)
        print(f"Final answer: {answer}")
        return {
            "query": question,
            "retrieved_documents": docs,
            "answer": answer,
        }

    return RunnableLambda(retrieve_documents) | RunnableLambda(finalize_answer)


def build_rag_agent(provider_config:ModelConfig):
    """Return an agent-style wrapper that fails gracefully when no vector data exists."""

    def agent(payload: dict[str, Any]) -> dict[str, Any]:
        question = payload["query"]
        vector_store = get_vector_store(provider_config)

        if _collection_count(vector_store) == 0:
            return {
                "query": question,
                "status": "no_data",
                "answer": "No matching data found in the vector store. Index a PDF first before asking questions.",
                "retrieved_documents": [],
            }

        docs = vector_store.similarity_search_with_score(question, k=5)
        return {
            "query": question,
            "status": "ok",
            "answer": _build_rag_answer(provider_config,question, docs),
            "retrieved_documents": docs,
        }

    return RunnableLambda(agent)
