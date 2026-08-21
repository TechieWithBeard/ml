from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from techiewithbeard_ai.job_match.nodes import (
    analyze_transferability,
    calculate_score,
    generate_critique,
    match_skills,
    parse_requirements,
    parse_resume,
)
from techiewithbeard_ai.job_match.state import JobMatchState

def check_missing_skills(state: JobMatchState) -> str:

    missing_skills = state.get("missing_skills") or []

    if missing_skills:
        return "transferability"

    return "scoring"

def build_job_match_graph():

    graph = StateGraph(JobMatchState)

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

    graph.add_edge(
        START,
        "parse_resume",
    )

    graph.add_edge(
        "parse_resume",
        "parse_requirements",
    )
    
    graph.add_edge(
        "parse_requirements",
        "match_skills",
    )
    
    graph.add_conditional_edges(
        "match_skills",
        check_missing_skills,
        {
            "transferability": "transferability",
            "scoring": "scoring",
        },
    )

    graph.add_edge(
        "transferability",
        "scoring",
    )

    graph.add_edge(
        "scoring",
        "critique",
    )

    graph.add_edge(
        "critique",
        END,
    )
    
    return graph.compile()



