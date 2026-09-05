import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr

from techiewithbeard_ai.schema.provider import ModelConfig
from techiewithbeard_ai.portfolio_agent.discovery import discover_mcp_tools
from techiewithbeard_ai.portfolio_agent.graph import build_portfolio_agent_graph

app = FastAPI(
    title="Vishnu Thankappan Portfolio LangGraph Agent",
    description="Token-efficient LangGraph multi-agent service powering MCP & portfolio queries.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import time

MAX_FREE_PROMPTS = 3
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def check_and_update_session_quota(session_id: str, has_custom_auth: bool) -> tuple[bool, int]:
    """
    Returns (is_allowed, quota_remaining).
    If user provided custom credentials (OpenAI key or Hugging Face token), unlimited queries are granted (-1).
    """
    if has_custom_auth:
        return True, -1

    now = time.time()
    # Expire entries older than 24 hours to prevent memory creep
    for s in list(SESSION_STORE.keys()):
        if now - SESSION_STORE[s].get("timestamp", 0) > 86400:
            SESSION_STORE.pop(s, None)

    sid = session_id or "anonymous"
    entry = SESSION_STORE.setdefault(sid, {"prompts": 0, "timestamp": now})
    entry["timestamp"] = now

    if entry["prompts"] >= MAX_FREE_PROMPTS:
        return False, 0

    entry["prompts"] += 1
    remaining = MAX_FREE_PROMPTS - entry["prompts"]
    return True, remaining


class QueryRequest(BaseModel):
    question: str
    target_url: Optional[str] = "https://www.techiewithbeard.com"
    provider: Optional[str] = "openai"  # "openai", "hugging face", "ollama"
    chat_model: Optional[str] = None
    openai_base_url: Optional[str] = None
    ollama_url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    references: List[str] = []
    selected_tool: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thought_trace: List[str] = []
    quota_remaining: Optional[int] = None
    is_free_tier: bool = False
    requires_custom_key: bool = False


import gradio as gr
from techiewithbeard_ai.portfolio_agent.ui import create_gradio_ui


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "portfolio-langgraph-agent",
        "environment": os.environ.get("RENDER", "local"),
    }


@app.post("/agent/query", response_model=QueryResponse)
async def query_langgraph_agent(
    payload: QueryRequest,
    x_openai_key: Optional[str] = Header(None, alias="x-openai-key"),
    x_hf_token: Optional[str] = Header(None, alias="x-hf-token"),
    x_session_id: Optional[str] = Header(None, alias="x-session-id"),
    authorization: Optional[str] = Header(None),
):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    target_url = payload.target_url or "https://www.techiewithbeard.com"

    # Determine custom credentials vs shared key
    custom_openai_key = x_openai_key or (
        authorization.replace("Bearer ", "") if authorization and "Bearer " in authorization else None
    )
    custom_hf_token = x_hf_token

    has_custom_auth = bool(custom_openai_key or custom_hf_token)
    session_id = x_session_id or "anonymous"

    # Check freemium session quota (3 prompts free on shared demo key)
    is_allowed, quota_remaining = check_and_update_session_quota(session_id, has_custom_auth)
    if not is_allowed:
        return QueryResponse(
            answer=(
                "⚡ **Demo Quota Reached (3/3 Free Prompts Used)**\n\n"
                "You have used your 3 free prompts on the shared demo server! "
                "To continue querying Vishnu's AI agent without limits:\n\n"
                "1. Click **Settings (⚙️)** in the top bar.\n"
                "2. Provide your personal **OpenAI API Key** (`sk-...`) or **Hugging Face Token** (`hf_...`).\n"
                "3. Your credentials remain private in your browser session and unlock **unlimited prompts**."
            ),
            references=[f"{target_url.rstrip('/')}/experience", f"{target_url.rstrip('/')}/demos"],
            quota_remaining=0,
            is_free_tier=True,
            requires_custom_key=True,
        )

    provider = (payload.provider or "openai").lower()
    # On production, default to openai if an unroutable local provider was chosen
    if os.environ.get("RENDER") and provider not in ["openai", "hugging face", "huggingface"]:
        provider = "openai"

    api_key_str = custom_openai_key or os.environ.get("OPENAI_API_KEY")
    hf_token_str = custom_hf_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    config = ModelConfig(
        provider=provider,
        chat_model=payload.chat_model or ("gpt-4o-mini" if provider == "openai" else "Qwen/Qwen2.5-7B-Instruct"),
        openai_api_key=SecretStr(api_key_str) if api_key_str else None,
        openai_base_url=payload.openai_base_url,
        hf_token=SecretStr(hf_token_str) if hf_token_str else None,
        ollama_url=payload.ollama_url or "http://localhost:11434",
        temperature=0.1,
        max_new_tokens=512,
    )

    try:
        tools = discover_mcp_tools(target_url)
        graph = build_portfolio_agent_graph(config, target_url)

        initial_state = {
            "question": question,
            "target_url": target_url,
            "tools": tools,
            "selected_tool": None,
            "tool_args": {},
            "raw_data": None,
            "pruned_data": None,
            "final_answer": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thought_trace": [f"Target URL: {target_url}", f"Provider: {provider}"],
        }

        result = graph.invoke(initial_state)

        p_tokens = result.get("prompt_tokens", 0)
        c_tokens = result.get("completion_tokens", 0)

        references = [f"{target_url.rstrip('/')}/experience", f"{target_url.rstrip('/')}/demos"]

        return QueryResponse(
            answer=result.get("final_answer", "Agent executed."),
            references=references,
            selected_tool=result.get("selected_tool"),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
            thought_trace=result.get("thought_trace", []),
            quota_remaining=quota_remaining,
            is_free_tier=not has_custom_auth,
            requires_custom_key=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent graph execution failed: {str(e)}")


# Mount Gradio interactive cockpit at root
gradio_demo = create_gradio_ui()
app = gr.mount_gradio_app(app, gradio_demo, path="/")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("techiewithbeard_ai.portfolio_agent.server:app", host="0.0.0.0", port=port, reload=True)
