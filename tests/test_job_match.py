from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from techiewithbeard_ai.job_match.personality import ResumePersonality, get_personality_config
from techiewithbeard_ai.job_match.personality_critique import generate_personality_critique
from techiewithbeard_ai.job_match.experience_agents import (
    tailor_experience_section,
    skill_to_experience_orchestrator,
)
from techiewithbeard_ai.job_match.schemas import Experience
from techiewithbeard_ai.schema.provider import ModelConfig


def test_personality_critique_prompt_and_execution(monkeypatch):
    critique_json = (
        '{"strengths": ["Strong Python"], "weaknesses": ["Needs Docker"], '
        '"missing_skills": ["Docker"], "learning_potential": "High", '
        '"recommendations": ["Learn Docker"], "overall_assessment": "Good fit"}'
    )

    mock_llm = RunnableLambda(lambda x: AIMessage(content=critique_json))
    monkeypatch.setattr(
        "techiewithbeard_ai.job_match.personality_critique.get_chat_model",
        lambda cfg: mock_llm,
    )

    config = ModelConfig(provider="ollama", model="llama3")
    critique = generate_personality_critique(
        candidate_name="Alice",
        candidate_skills=["Python"],
        experience_summaries="- Software Engineer at Acme",
        required_skills=["Python", "Docker"],
        matching_skills=["Python"],
        missing_skills=["Docker"],
        personality=ResumePersonality.AMERICAN,
        config=config,
    )

    assert critique.strengths == ["Strong Python"]
    assert critique.missing_skills == ["Docker"]
    assert critique.overall_assessment == "Good fit"


def test_tailor_experience_prompt_and_execution(monkeypatch):
    mock_llm = RunnableLambda(
        lambda x: AIMessage(content='["Engineered high-scale Python services"]')
    )
    monkeypatch.setattr(
        "techiewithbeard_ai.job_match.experience_agents.get_chat_model",
        lambda cfg: mock_llm,
    )

    config = ModelConfig(provider="ollama", model="llama3")
    exp = Experience(
        company="Acme",
        title="Software Engineer",
        start_date="2020",
        end_date="2023",
        bullets=["Built Python services"],
    )

    tailored = tailor_experience_section(
        experience=exp,
        required_skills=["Python"],
        relevant_skills_for_this_exp=["Python"],
        personality=ResumePersonality.TECH_STARTUP,
        config=config,
    )

    assert tailored.bullets == ["Engineered high-scale Python services"]
