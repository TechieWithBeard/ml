from techiewithbeard_ai.schema.provider import ModelConfig
from techiewithbeard_ai.chains.rag_chain import ResumeAnalyser, build_rag_chain, build_rag_agent


class FakeStore:
    def __init__(self, count: int = 2):
        self._count = count

    def count(self) -> int:
        return self._count

    def similarity_search_with_score(self, question: str, k: int = 5):
        return [
            (type("Doc", (), {"page_content": "doc content", "metadata": {"source": "pdf"}})(), 0.1),
        ]


def test_build_rag_chain_returns_retrieval_payload(monkeypatch):
    monkeypatch.setattr(
        "techiewithbeard_ai.chains.rag_chain.get_vector_store",
        lambda *args, **kwargs: FakeStore(count=2),
    )
    fake_analyser = ResumeAnalyser(query="What is the candidate name?", observation="answer")
    monkeypatch.setattr(
        "techiewithbeard_ai.chains.rag_chain._build_rag_answer_with_pydantic",
        lambda *args, **kwargs: fake_analyser,
    )

    config = ModelConfig(provider="ollama", model="llama3")
    chain = build_rag_chain(config)
    result = chain.invoke({"query": "What is the candidate name?"})

    assert result["query"] == "What is the candidate name?"
    assert result["answer"].observation == "answer"
    assert len(result["retrieved_documents"]) == 1


def test_build_rag_agent_returns_fallback_when_store_empty(monkeypatch):
    monkeypatch.setattr(
        "techiewithbeard_ai.chains.rag_chain.get_vector_store",
        lambda *args, **kwargs: FakeStore(count=0),
    )

    config = ModelConfig(provider="ollama", model="llama3")
    agent = build_rag_agent(config)
    result = agent.invoke({"query": "What is the candidate name?"})

    assert result["query"] == "What is the candidate name?"
    assert result["status"] == "no_data"
    assert "No matching data" in result["answer"]
