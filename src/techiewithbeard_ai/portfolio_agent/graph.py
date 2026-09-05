import json
import re
import requests
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage

from techiewithbeard_ai.schema.provider import ModelConfig
from techiewithbeard_ai.agents.agents import get_chat_model
from techiewithbeard_ai.portfolio_agent.discovery import ToolDescriptor, discover_mcp_tools, format_tools_for_prompt


class PortfolioAgentState(TypedDict):
    question: str
    target_url: str
    tools: List[ToolDescriptor]
    selected_tool: Optional[str]
    tool_args: Dict[str, Any]
    raw_data: Any
    pruned_data: Any
    final_answer: str
    prompt_tokens: int
    completion_tokens: int
    thought_trace: List[str]


def create_router_node(model):
    def route_intent(state: PortfolioAgentState) -> Dict[str, Any]:
        tools_summary = format_tools_for_prompt(state["tools"])
        user_q = state["question"].strip()

        system_prompt = (
            "You are a lightweight intent router for Vishnu Thankappan's portfolio.\n"
            "Given the user's question, choose the single best tool from this catalog:\n"
            f"{tools_summary}\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            '{"tool": "tool_name_or_null", "args": {"param_key": "param_value"}}\n'
            "If the question is a general greeting or does not need a tool, set tool to null."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_q),
        ]

        response = model.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Approximate token usage
        prompt_tokens = len(system_prompt.split()) + len(user_q.split())
        completion_tokens = len(content.split())

        tool_name = None
        tool_args: Dict[str, Any] = {}

        try:
            # Extract JSON cleanly from response
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                tool_name = parsed.get("tool")
                tool_args = parsed.get("args", {})
        except Exception:
            # Fallback heuristics if router fails
            q_lower = user_q.lower()
            if any(k in q_lower for k in ["aveva", "experience", "work", "history", "career"]):
                tool_name = "get_work_history"
            elif any(k in q_lower for k in ["skill", "angular", "react", "langgraph"]):
                tool_name = "search_skills"
            elif any(k in q_lower for k in ["demo", "project", "app"]):
                tool_name = "get_live_demos"
            elif any(k in q_lower for k in ["who", "about", "contact", "email"]):
                tool_name = "get_architect_profile"
            else:
                tool_name = "ask_portfolio_agent"
                tool_args = {"question": user_q}

        trace = list(state.get("thought_trace", []))
        trace.append(f"Router selected tool: {tool_name} with args: {tool_args}")

        return {
            "selected_tool": tool_name,
            "tool_args": tool_args,
            "prompt_tokens": state.get("prompt_tokens", 0) + prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + completion_tokens,
            "thought_trace": trace,
        }

    return route_intent


def execute_tool_node(state: PortfolioAgentState) -> Dict[str, Any]:
    tool = state.get("selected_tool")
    args = state.get("tool_args", {})
    target_url = state.get("target_url", "https://www.techiewithbeard.com").rstrip("/")
    if not target_url.startswith("http"):
        target_url = f"https://{target_url}"

    raw_data: Any = None
    trace = list(state.get("thought_trace", []))

    if not tool:
        trace.append("No tool execution required (direct synthesis).")
        return {"raw_data": None, "thought_trace": trace}

    try:
        # Route execution to appropriate API path
        if tool == "get_work_history":
            company = args.get("company", "")
            endpoint = f"{target_url}/api/experience"
            resp = requests.get(endpoint, timeout=5)
            if resp.status_code == 200:
                raw_data = resp.json()
                if company:
                    c_lower = company.lower()
                    raw_data = [e for e in raw_data if c_lower in e.get("company", "").lower()]
            else:
                # Fallback to direct query
                resp = requests.post(f"{target_url}/api/agent/query", json={"question": state["question"]}, timeout=5)
                raw_data = resp.json() if resp.status_code == 200 else None

        elif tool == "search_skills":
            keyword = args.get("keyword", "")
            resp = requests.get(f"{target_url}/api/skills", timeout=5)
            if resp.status_code == 200:
                raw_data = resp.json()
            else:
                resp = requests.post(f"{target_url}/api/agent/query", json={"question": state["question"]}, timeout=5)
                raw_data = resp.json() if resp.status_code == 200 else None

        elif tool == "get_live_demos":
            resp = requests.get(f"{target_url}/api/demos", timeout=5)
            raw_data = resp.json() if resp.status_code == 200 else None

        elif tool == "get_architect_profile":
            resp = requests.get(f"{target_url}/api/profile", timeout=5)
            raw_data = resp.json() if resp.status_code == 200 else None

        else:
            # Default to /api/agent/query
            q = args.get("question") or state["question"]
            resp = requests.post(f"{target_url}/api/agent/query", json={"question": q}, timeout=5)
            raw_data = resp.json() if resp.status_code == 200 else None

        trace.append(f"HTTP call to {target_url} succeeded. Fetched payload size: {len(str(raw_data))} chars.")
    except Exception as e:
        trace.append(f"HTTP call failed ({e}). Utilizing fallback.")
        raw_data = None

    return {"raw_data": raw_data, "thought_trace": trace}


def prune_data_node(state: PortfolioAgentState) -> Dict[str, Any]:
    """
    Prune unneeded keys (UUIDs, timestamps, nulls) to keep LLM context tokens minimal.
    """
    data = state.get("raw_data")
    trace = list(state.get("thought_trace", []))

    if data is None:
        return {"pruned_data": None, "thought_trace": trace}

    def clean(item):
        if isinstance(item, dict):
            return {
                k: clean(v)
                for k, v in item.items()
                if k not in ["id", "created_at", "updated_at", "schema"] and v is not None and v != ""
            }
        if isinstance(item, list):
            return [clean(x) for x in item[:6]]  # Cap list to top 6 items
        return item

    pruned = clean(data)
    trace.append(f"Data pruned for token efficiency. Payload compressed to {len(str(pruned))} chars.")
    return {"pruned_data": pruned, "thought_trace": trace}


def create_synthesizer_node(model):
    def synthesize_answer(state: PortfolioAgentState) -> Dict[str, Any]:
        user_q = state["question"]
        pruned = state.get("pruned_data")
        trace = list(state.get("thought_trace", []))

        data_block = json.dumps(pruned, indent=1) if pruned else "No external data fetched; answer conversationally."

        system_prompt = (
            "You are the official AI representative for Vishnu Thankappan (@techiewithbeard), "
            "a Senior Frontend Engineer & UI/AI Architect with 7+ years enterprise experience.\n"
            "Answer the question directly, authoritatively, and professionally using the verified facts below.\n"
            "Highlight metrics (e.g. 30% build time cut, 42% bundle reduction, Angular 22 Signals, Nx Monorepos).\n"
            "Include markdown links where available. Keep answer concise and high-impact (under 180 words).\n\n"
            f"VERIFIED DATA:\n{data_block}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_q),
        ]

        response = model.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        prompt_tokens = len(system_prompt.split()) + len(user_q.split())
        completion_tokens = len(answer.split())

        trace.append(f"Synthesizer completed answer ({len(answer)} chars).")

        return {
            "final_answer": answer,
            "prompt_tokens": state.get("prompt_tokens", 0) + prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + completion_tokens,
            "thought_trace": trace,
        }

    return synthesize_answer


def build_portfolio_agent_graph(config: ModelConfig, target_url: str):
    """
    Constructs the token-efficient LangGraph State Machine.
    """
    model = get_chat_model(config)

    graph = StateGraph(PortfolioAgentState)

    graph.add_node("route_intent", create_router_node(model))
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("prune_data", prune_data_node)
    graph.add_node("synthesize_answer", create_synthesizer_node(model))

    graph.add_edge(START, "route_intent")
    graph.add_edge("route_intent", "execute_tool")
    graph.add_edge("execute_tool", "prune_data")
    graph.add_edge("prune_data", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)

    return graph.compile()
