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


class QueryRequest(BaseModel):
    question: str
    target_url: Optional[str] = "https://www.techiewithbeard.com"
    provider: Optional[str] = "openai"  # "ollama", "openai", "hugging face"
    chat_model: Optional[str] = None
    openai_base_url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    references: List[str] = []
    selected_tool: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thought_trace: List[str] = []


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
    authorization: Optional[str] = Header(None),
):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    target_url = payload.target_url or "https://www.techiewithbeard.com"

    # Resolve API key from header or environment
    api_key_str = x_openai_key or (authorization.replace("Bearer ", "") if authorization else None)
    if not api_key_str:
        api_key_str = os.environ.get("OPENAI_API_KEY")

    provider = payload.provider or ("openai" if api_key_str else "ollama")

    config = ModelConfig(
        provider=provider,
        chat_model=payload.chat_model or ("gpt-4o-mini" if provider == "openai" else "gemma4:e4b"),
        openai_api_key=SecretStr(api_key_str) if api_key_str else None,
        openai_base_url=payload.openai_base_url,
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
