from typing import TypedDict
from techiewithbeard_ai.job_match.schemas import Experience, ResumeDocument, ResumeTailoring, SkillMatch, Transferability, Critique
from techiewithbeard_ai.schema.provider import ModelConfig

class JobMatchState(TypedDict, total=False):

    # Input
    resume_text: str
    resume_document: ResumeDocument | None
    resume_tailoring: ResumeTailoring | None
    tailored_resume: ResumeDocument | None
    resume_html: str
    job_description: str
    config: ModelConfig

    # Resume
    candidate_name: str | None
    candidate_skills: list[str]
    candidate_experience: list[Experience]

    # Job requirements
    required_skills: list[str]
    required_experience: list[str]
    responsibilities: list[str]

    # Matching
    matching_skills: list[str]
    missing_skills: list[str]
    skill_matches: list[SkillMatch]

    # Transferability
    transferability: list[Transferability]

    # Scoring
    skill_score: float
    experience_score: float
    responsibility_score: float
    overall_score: float

    # Critique
    critique: Critique

    # Final
    final_report: dict