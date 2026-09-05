import requests
from typing import Any, Dict, List
from pydantic import BaseModel


class ToolDescriptor(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = {}
    endpoint: str | None = None


DEFAULT_FALLBACK_TOOLS = [
    ToolDescriptor(
        name="get_architect_profile",
        description="Retrieve Vishnu Thankappan's architect profile, contact details, availability status, and bio.",
        parameters={},
        endpoint="/api/profile",
    ),
    ToolDescriptor(
        name="get_work_history",
        description="Retrieve enterprise work history and track record at AVEVA, Maistering B.V, and ACI Logistix.",
        parameters={"company": "string (optional filter e.g. 'AVEVA', 'Maistering')"},
        endpoint="/api/experience",
    ),
    ToolDescriptor(
        name="search_skills",
        description="Search verified technical skills across Frontend Architecture, AI Interfaces, and DevOps.",
        parameters={"keyword": "string filter e.g. 'Angular', 'LangGraph', 'TypeScript'"},
        endpoint="/api/skills",
    ),
    ToolDescriptor(
        name="get_live_demos",
        description="Retrieve all interactive AI and frontend demos with deep-links, live sandboxes, and documentation.",
        parameters={},
        endpoint="/api/demos",
    ),
    ToolDescriptor(
        name="ask_portfolio_agent",
        description="Query Vishnu's career, architectural decisions, and projects using natural language.",
        parameters={"question": "string natural language query"},
        endpoint="/api/agent/query",
    ),
]


def discover_mcp_tools(target_url: str, timeout_seconds: int = 5) -> List[ToolDescriptor]:
    """
    Dynamically discover Model Context Protocol tools from a given web URL.
    Checks:
    1. /.well-known/mcp.json
    2. /mcp.json
    3. /api/agent/tools
    Falls back to built-in portfolio tools if unreachable.
    """
    url = target_url.strip().rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"

    discovery_candidates = [
        f"{url}/.well-known/mcp.json",
        f"{url}/mcp.json",
        f"{url}/api/agent/tools",
    ]

    for endpoint in discovery_candidates:
        try:
            resp = requests.get(endpoint, timeout=timeout_seconds, headers={"User-Agent": "PortfolioMcpAgent/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                tools: List[ToolDescriptor] = []

                # Format 1: Standard .well-known/mcp.json with webmcp.tools
                if "webmcp" in data and isinstance(data["webmcp"], dict) and "tools" in data["webmcp"]:
                    for t in data["webmcp"]["tools"]:
                        if isinstance(t, str):
                            tools.append(ToolDescriptor(name=t, description=f"Tool: {t}"))
                        elif isinstance(t, dict):
                            tools.append(
                                ToolDescriptor(
                                    name=t.get("name", "unknown_tool"),
                                    description=t.get("description", ""),
                                    parameters=t.get("inputSchema", {}).get("properties", {}),
                                )
                            )
                    if tools:
                        return tools

                # Format 2: Standard MCP tools array
                if "tools" in data and isinstance(data["tools"], list):
                    for t in data["tools"]:
                        tools.append(
                            ToolDescriptor(
                                name=t.get("name", "unknown_tool"),
                                description=t.get("description", ""),
                                parameters=t.get("inputSchema", {}).get("properties", {}),
                            )
                        )
                    if tools:
                        return tools
        except Exception:
            continue

    return DEFAULT_FALLBACK_TOOLS


def format_tools_for_prompt(tools: List[ToolDescriptor]) -> str:
    """
    Format tools into a super-compact string to minimize prompt tokens.
    Uses ~15-20 tokens per tool.
    """
    lines = []
    for t in tools:
        param_desc = ", ".join(f"{k}: {v}" for k, v in t.parameters.items()) if t.parameters else "none"
        lines.append(f"- {t.name}: {t.description} (Params: {param_desc})")
    return "\n".join(lines)
