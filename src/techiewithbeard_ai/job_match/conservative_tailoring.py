"""
Conservative resume tailoring strategy.

Enhances existing resume content based on job requirements
WITHOUT inventing or adding false data.

Core principle:
- Only emphasize what already exists
- Never add skills/technologies not listed
- Never invent achievements
- Only reorder/rephrase to highlight relevance
"""

from techiewithbeard_ai.job_match.schemas import ResumeDocument


def get_relevant_skills_from_resume(
    resume: ResumeDocument,
    required_skills: list[str],
) -> set[str]:
    """
    Find which required skills are already in the resume.
    
    Returns a set of skills that exist in the resume
    and match required skills (case-insensitive).
    """
    candidate_skill_set = {
        skill.strip().lower()
        for group in resume.skills
        for skill in group.skills
    }
    
    required_skills_lower = {
        skill.strip().lower()
        for skill in required_skills
    }
    
    # Intersection = skills that appear in both
    return candidate_skill_set & required_skills_lower


def get_relevant_experience_keywords(
    resume: ResumeDocument,
    required_skills: list[str],
    responsibilities: list[str],
) -> set[str]:
    """
    Extract keywords from resume that match job requirements.
    
    Looks for words in experience bullets that match
    required skills or responsibility keywords.
    """
    all_keywords = set()
    
    # Add required skills
    all_keywords.update(
        skill.lower() for skill in required_skills
    )
    
    # Add responsibility keywords (first few words)
    for resp in responsibilities:
        words = resp.lower().split()
        all_keywords.update(words[:5])  # First 5 words per responsibility
    
    return all_keywords


def has_relevant_experience(
    resume: ResumeDocument,
    required_experience_keywords: list[str],
) -> bool:
    """
    Check if resume contains bullets with relevant keywords.
    
    Returns True if at least one experience bullet contains
    one or more required keywords.
    """
    experience_text = " ".join(
        bullet.lower()
        for exp in resume.experience
        for bullet in exp.bullets
    )
    
    for keyword in required_experience_keywords:
        if keyword.lower() in experience_text:
            return True
    
    return False


def identify_matching_skill_groups(
    resume: ResumeDocument,
    required_skills: list[str],
) -> list[str]:
    """
    Identify skill group categories that are relevant to the job.
    
    Returns list of skill group category names that contain
    at least one required skill.
    """
    relevant_groups = []
    required_lower = {s.lower() for s in required_skills}
    
    for group in resume.skills:
        group_skills_lower = {s.lower() for s in group.skills}
        
        if group_skills_lower & required_lower:
            relevant_groups.append(group.category)
    
    return relevant_groups


def get_recommended_skill_group_order(
    resume: ResumeDocument,
    required_skills: list[str],
) -> list[str]:
    """
    Recommend skill group ordering based on job relevance.
    
    Places skill groups that contain required skills first.
    """
    relevant_groups = identify_matching_skill_groups(
        resume,
        required_skills,
    )
    
    # All groups
    all_groups = [group.category for group in resume.skills]
    
    # Relevant groups first, then others
    reordered = relevant_groups + [
        g for g in all_groups if g not in relevant_groups
    ]
    
    return reordered
