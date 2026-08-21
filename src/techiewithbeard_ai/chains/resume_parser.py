
from typing import cast

from techiewithbeard_ai.schema.provider import ModelConfig
from techiewithbeard_ai.job_match.schemas import ResumeProfile
from techiewithbeard_ai.job_match.state import JobMatchState
from techiewithbeard_ai.agents.agents import get_chat_model
from langchain_core.prompts import ChatPromptTemplate

def check_missing_skills(state: JobMatchState):

    if state.get("missing_skills"):
        return "transferability"

    return "scoring"

def parse_resume(
    conf: ModelConfig,
    state: JobMatchState,
) -> dict:

    resume_text = state.get("resume_text")

    if resume_text is None:
        raise ValueError("Resume text is missing from graph state.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a resume analyzer.

                    Extract information ONLY from the provided resume.

                    Return:
                    - candidate_name
                    - technical skills
                    - work experience

                    Rules:
                    - Do not infer skills that are not explicitly mentioned.
                    - Do not invent experience.
                    - Return an empty list when information is not available.
                    """,
            ),
            (
                "human",
                """
                RESUME:

                {resume_text}
                """,
            ),
        ]
    )

    llm = get_chat_model(conf).with_structured_output(
        ResumeProfile
    )

    chain = prompt | llm

    raw_result = chain.invoke(
        {
            "resume_text": resume_text,
        }
    )

    result = cast(ResumeProfile, raw_result)

    return {
        "candidate_name": result.candidate_name,
        "candidate_skills": result.skills,
        "candidate_experience": result.experience,
    }
