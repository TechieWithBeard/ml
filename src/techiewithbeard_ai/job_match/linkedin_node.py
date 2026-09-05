"""
LinkedIn job extraction node for the LangGraph workflow.
"""

from langsmith import traceable

from techiewithbeard_ai.job_match.linkedin_extractor import extract_linkedin_job_sync
from techiewithbeard_ai.job_match.state import JobMatchState


@traceable
def extract_linkedin_job_node(
    state: JobMatchState,
) -> dict:
    """
    Extract job requirements from LinkedIn URL.
    
    Input state:
    - linkedin_url: The LinkedIn job posting URL
    - config: Model configuration
    
    Output:
    - job_description: Full job description text
    - required_skills: Extracted skills
    - required_experience: Extracted experience
    - responsibilities: Extracted responsibilities
    """
    
    linkedin_url = state.get("linkedin_url")
    config = state.get("config")
    
    if not linkedin_url:
        raise ValueError(
            "LinkedIn URL is required for extraction."
        )
    
    if config is None:
        raise ValueError(
            "Model configuration is missing."
        )
    
    print("\n========== LINKEDIN EXTRACTION START ==========")
    print(f"Extracting from: {linkedin_url}")
    
    result = extract_linkedin_job_sync(linkedin_url, config)
    
    if not result.get("success"):
        raise ValueError(
            f"Failed to extract from LinkedIn: {result.get('error')}"
        )
    
    job_text = result.get("job_text", "")
    
    print(f"Extracted {len(job_text)} characters")
    print("=========== LINKEDIN EXTRACTION END ===========\n")
    
    # Return the full text as job_description
    # The rest of the pipeline will parse requirements as normal
    return {
        "job_description": job_text,
        "linkedin_source": linkedin_url,
    }
