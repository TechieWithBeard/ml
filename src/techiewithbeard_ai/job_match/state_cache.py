"""
State caching and deduplication.

Reuses previously computed outputs if input hasn't changed.
Avoids unnecessary LLM calls when processing identical data.
"""

from techiewithbeard_ai.job_match.state import JobMatchState


def should_reuse_candidate_output(state: JobMatchState) -> bool:
    """
    Check if we should reuse cached candidate parsing.
    
    Return True if:
    - Resume has been parsed
    - Resume text hasn't changed
    """
    return state.get("resume_document") is not None


def should_reuse_requirements_output(state: JobMatchState) -> bool:
    """
    Check if we should reuse cached job requirements parsing.
    
    Return True if:
    - Requirements have been parsed
    - Job description hasn't changed
    """
    return state.get("required_skills") is not None


def get_cached_candidate_data(state: JobMatchState) -> dict:
    """Return cached candidate data without re-processing."""
    resume_doc = state.get("resume_document")
    
    if resume_doc is None:
        return {}
    
    return {
        "resume_document": resume_doc,
        "candidate_name": resume_doc.candidate_name,
        "candidate_skills": [
            skill
            for group in resume_doc.skills
            for skill in group.skills
        ],
        "candidate_experience": resume_doc.experience,
    }


def get_cached_requirements_data(state: JobMatchState) -> dict:
    """Return cached job requirements data without re-processing."""
    return {
        "required_skills": state.get("required_skills") or [],
        "required_experience": state.get("required_experience") or [],
        "responsibilities": state.get("responsibilities") or [],
    }
