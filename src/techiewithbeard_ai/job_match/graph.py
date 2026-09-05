from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from techiewithbeard_ai.job_match.linkedin_node import extract_linkedin_job_node
from techiewithbeard_ai.job_match.nodes import (
    analyze_transferability,
    calculate_score,
    generate_personality_critique_node,
    match_skills,
    parse_requirements,
    parse_resume,
    tailor_resume_with_experience_agents,
)
from techiewithbeard_ai.job_match.state import JobMatchState


def check_missing_skills(
    state: JobMatchState,
) -> str:

    missing_skills = state.get("missing_skills") or []

    if missing_skills:
        return "transferability"

    return "scoring"


def route_job_source(
    state: JobMatchState,
) -> str:
    """
    Route between LinkedIn extraction and manual job description.
    """
    linkedin_url = state.get("linkedin_url")
    
    if linkedin_url and isinstance(linkedin_url, str) and linkedin_url.strip():
        return "extract_linkedin_job"
    
    return "parse_requirements"


def build_job_match_graph():

    graph = StateGraph(JobMatchState)

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    graph.add_node(
        "extract_linkedin_job",
        extract_linkedin_job_node,
    )

    graph.add_node(
        "parse_resume",
        parse_resume,
    )

    graph.add_node(
        "parse_requirements",
        parse_requirements,
    )

    graph.add_node(
        "match_skills",
        match_skills,
    )

    graph.add_node(
        "transferability",
        analyze_transferability,
    )

    graph.add_node(
        "scoring",
        calculate_score,
    )

    graph.add_node(
        "critique",
        generate_personality_critique_node,
    )

    # graph.add_node(
    #     "generate_resume_html",
    #     generate_resume_html,
    # )
    
    graph.add_node(
        "tailor_resume",
        tailor_resume_with_experience_agents,
    )

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    graph.add_edge(
        START,
        "parse_resume",
    )

    # Route between LinkedIn extraction and manual job description
    graph.add_conditional_edges(
        START,
        route_job_source,
        {
            "extract_linkedin_job": "extract_linkedin_job",
            "parse_requirements": "parse_requirements",
        },
    )

    # LinkedIn extraction feeds into parse_requirements
    graph.add_edge(
        "extract_linkedin_job",
        "parse_requirements",
    )

    # ---------------------------------------------------------
    # Resume + Job Requirements
    # ---------------------------------------------------------

    graph.add_edge(
        "parse_resume",
        "match_skills",
    )

    graph.add_edge(
        "parse_requirements",
        "match_skills",
    )

    # ---------------------------------------------------------
    # Skill Matching
    # ---------------------------------------------------------

    graph.add_conditional_edges(
        "match_skills",
        check_missing_skills,
        {
            "transferability": "transferability",
            "scoring": "scoring",
        },
    )

    # ---------------------------------------------------------
    # Transferability
    # ---------------------------------------------------------

    graph.add_edge(
        "transferability",
        "scoring",
    )

    # ---------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------

    graph.add_edge(
        "scoring",
        "critique",
    )

    # ---------------------------------------------------------
    # AI Critique
    # ---------------------------------------------------------

    # graph.add_edge(
    #     "critique",
    #     "generate_resume_html",
    # )
    
    graph.add_edge(
        "critique",
        "tailor_resume",
    )

    graph.add_edge(
        "tailor_resume",
        END,
    )
    
    # ---------------------------------------------------------
    # Resume HTML
    # ---------------------------------------------------------

    # graph.add_edge(
    #     "generate_resume_html",
    #     END,
    # )

    return graph.compile()
