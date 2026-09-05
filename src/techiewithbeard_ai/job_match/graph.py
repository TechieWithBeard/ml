from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from techiewithbeard_ai.job_match.nodes import (
    analyze_transferability,
    apply_resume_tailoring,
    apply_tailoring,
    calculate_score,
    generate_critique,
    match_skills,
    parse_requirements,
    parse_resume,
    tailor_resume,
)

from techiewithbeard_ai.job_match.state import JobMatchState


def check_missing_skills(
    state: JobMatchState,
) -> str:

    missing_skills = state.get("missing_skills") or []

    if missing_skills:
        return "transferability"

    return "scoring"


def build_job_match_graph():

    graph = StateGraph(JobMatchState)

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

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
        generate_critique,
    )

    # graph.add_node(
    #     "generate_resume_html",
    #     generate_resume_html,
    # )
    
    graph.add_node(
    "tailor_resume",
    tailor_resume,
    )

    graph.add_node(
        "apply_tailoring",
        apply_tailoring,
    )

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    graph.add_edge(
        START,
        "parse_resume",
    )

    graph.add_edge(
        START,
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
        "apply_tailoring",
    )

    graph.add_edge(
        "apply_tailoring",
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