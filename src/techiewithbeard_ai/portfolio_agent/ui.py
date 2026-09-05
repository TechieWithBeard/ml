import os
import json
import time
import gradio as gr
from pydantic import SecretStr

from techiewithbeard_ai.schema.provider import ModelConfig
from techiewithbeard_ai.portfolio_agent.discovery import discover_mcp_tools
from techiewithbeard_ai.portfolio_agent.graph import build_portfolio_agent_graph

IS_DEPLOYED = bool(os.environ.get("RENDER") or os.environ.get("SPACE_ID"))
PROVIDER_CHOICES = (
    ["OpenAI / Compatible", "Hugging Face"]
    if IS_DEPLOYED
    else ["OpenAI / Compatible", "Hugging Face", "Local (Ollama)"]
)
DEFAULT_PROVIDER = "OpenAI / Compatible"
DEFAULT_TARGET_URL = "https://www.techiewithbeard.com"

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="#070a11",
    body_text_color="#f8fafc",
    block_background_fill="#0f172a",
    block_border_width="1px",
    block_border_color="#1e293b",
)


def on_discover_tools(target_url: str):
    if not target_url.strip():
        return "⚠️ Please provide a valid portfolio URL."
    tools = discover_mcp_tools(target_url)
    md = f"### 🛠️ Discovered {len(tools)} WebMCP / MCP Tools from `{target_url}`\n\n"
    for t in tools:
        md += f"- **`{t.name}`**: {t.description}\n"
    return md


def on_provider_change(provider: str):
    if provider == "OpenAI / Compatible":
        return (
            gr.update(visible=True, label="OpenAI API Key (Session-Stored)", placeholder="sk-... (Leave empty to use 3 free demo queries)"),
            gr.update(visible=True, value="https://api.openai.com/v1"),
            gr.update(value="gpt-4o-mini"),
        )
    elif provider == "Hugging Face":
        return (
            gr.update(visible=True, label="Hugging Face User Access Token", placeholder="hf_..."),
            gr.update(visible=False),
            gr.update(value="Qwen/Qwen2.5-7B-Instruct"),
        )
    else:  # Local (Ollama)
        return (
            gr.update(visible=False),
            gr.update(visible=True, value="http://localhost:11434"),
            gr.update(value="llama3.2"),
        )


def chat_with_agent(
    message: str,
    history: list,
    target_url: str,
    provider_choice: str,
    api_key: str,
    base_url: str,
    model_name: str,
    session_token_store: dict,
    single_turn: bool = False,
):
    if not message.strip():
        yield history, "", "Ready", "None", []
        return

    # STEP 1: Immediately render user message & decision status card
    current_history = [] if single_turn else list(history)
    current_history.append({"role": "user", "content": message})
    current_history.append({
        "role": "assistant",
        "content": "🧭 **Routing Query Intent**\n*Analyzing question and discovering available WebMCP tools...*",
    })

    thought_trace = [
        f"Incoming user query: '{message}'",
        f"Target endpoint: {target_url}",
        "Evaluating tools and routing intent...",
    ]
    trace_text = "\n".join(f"• {t}" for t in thought_trace)

    # Yield IMMEDIATELY so the user question appears without any delay
    yield (
        current_history,
        "",  # Instantly clears input box in UI
        "📊 Token Meter: Initializing...",
        "⚡ Tool: 🧭 Routing intent...",
        trace_text,
    )

    provider_map = {
        "Local (Ollama)": "ollama",
        "OpenAI / Compatible": "openai",
        "Hugging Face": "hugging face",
    }
    provider = provider_map.get(provider_choice, "ollama")

    resolved_token = api_key or session_token_store.get("api_key") or (
        os.environ.get("HUGGINGFACEHUB_API_TOKEN", "") if provider == "hugging face" else os.environ.get("OPENAI_API_KEY", "")
    )
    if api_key:
        session_token_store["api_key"] = api_key

    default_model = "gpt-4o-mini" if provider == "openai" else ("Qwen/Qwen2.5-7B-Instruct" if provider == "hugging face" else "llama3.2")

    config = ModelConfig(
        provider=provider,
        chat_model=model_name or default_model,
        openai_api_key=SecretStr(resolved_token) if provider == "openai" and resolved_token else None,
        openai_base_url=base_url if base_url and "openai.com" not in base_url else None,
        hf_token=SecretStr(resolved_token) if provider == "hugging face" and resolved_token else None,
        ollama_url=base_url if provider == "ollama" and base_url else "http://localhost:11434",
        temperature=0.1,
        max_new_tokens=512,
    )

    try:
        tools = discover_mcp_tools(target_url)
        tool_names = [f"`{t.name}`" for t in tools]
        thought_trace.append(f"Discovered {len(tools)} tools: {', '.join(tool_names[:5])}")
        trace_text = "\n".join(f"• {t}" for t in thought_trace)

        current_history[-1]["content"] = (
            f"🧭 **Routing Query Intent**\n"
            f"*Discovered {len(tools)} tools ({', '.join(tool_names[:3])}...). Classifying optimal tool with LangGraph router...*"
        )
        yield current_history, "", "📊 Token Meter: Routing intent...", "⚡ Tool: 🧭 Classifying...", trace_text

        graph = build_portfolio_agent_graph(config, target_url)

        initial_state = {
            "question": message,
            "target_url": target_url,
            "tools": tools,
            "selected_tool": None,
            "tool_args": {},
            "raw_data": None,
            "pruned_data": None,
            "final_answer": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thought_trace": thought_trace,
        }

        selected_tool = "Direct Synthesis"
        prompt_tokens = 0
        completion_tokens = 0

        # STEP 2: Stream decision making node-by-node in real time
        for event in graph.stream(initial_state):
            for node_name, node_output in event.items():
                if "thought_trace" in node_output:
                    thought_trace = node_output["thought_trace"]
                if "prompt_tokens" in node_output:
                    prompt_tokens = node_output["prompt_tokens"]
                if "completion_tokens" in node_output:
                    completion_tokens = node_output["completion_tokens"]

                trace_text = "\n".join(f"• {t}" for t in thought_trace)

                if node_name == "route_intent":
                    selected_tool = node_output.get("selected_tool") or "Direct Synthesis"
                    tool_args = node_output.get("tool_args", {})
                    args_str = f" with `{json.dumps(tool_args)}`" if tool_args else ""

                    current_history[-1]["content"] = (
                        f"🎯 **Decision Made: Selected `{selected_tool}`**{args_str}\n\n"
                        f"🌐 *Executing verified portfolio API endpoint...*"
                    )
                    yield (
                        current_history,
                        "",
                        f"📊 Token Meter: ~{prompt_tokens} tokens",
                        f"⚡ Selected: {selected_tool}",
                        trace_text,
                    )

                elif node_name == "execute_tool":
                    raw_data = node_output.get("raw_data")
                    data_chars = len(str(raw_data)) if raw_data is not None else 0
                    current_history[-1]["content"] = (
                        f"🌐 **API Tool Executed: `{selected_tool}`**\n\n"
                        f"Fetched {data_chars} characters of verified facts.\n"
                        f"✂️ *Pruning unneeded keys and timestamps to save tokens...*"
                    )
                    yield (
                        current_history,
                        "",
                        f"📊 Token Meter: ~{prompt_tokens} tokens",
                        f"🌐 Executed: {selected_tool}",
                        trace_text,
                    )

                elif node_name == "prune_data":
                    pruned_data = node_output.get("pruned_data")
                    pruned_chars = len(str(pruned_data)) if pruned_data is not None else 0
                    current_history[-1]["content"] = (
                        f"✂️ **Payload Pruned: Compressed to {pruned_chars} chars** (<120 tokens)\n\n"
                        f"✍️ *Synthesizing authoritative executive answer...*"
                    )
                    yield (
                        current_history,
                        "",
                        f"📊 Token Meter: ~{prompt_tokens} tokens",
                        f"✂️ Pruned Payload",
                        trace_text,
                    )

                elif node_name == "synthesize_answer":
                    final_ans = node_output.get("final_answer", "")
                    total_tokens = prompt_tokens + completion_tokens
                    token_summary = (
                        f"⚡ Stateless Query: Prompt ~{prompt_tokens} | "
                        f"Completion ~{completion_tokens} | "
                        f"Total ~{total_tokens} tokens (0 past conversation tokens)"
                    )

                    # Stream typing effect for the final answer
                    words = final_ans.split(" ")
                    chunk_step = 5
                    if len(words) > 8:
                        for i in range(chunk_step, len(words), chunk_step):
                            partial = " ".join(words[:i])
                            current_history[-1]["content"] = partial
                            yield (
                                current_history,
                                "",
                                token_summary,
                                f"✍️ Synthesizing ({selected_tool})",
                                trace_text,
                            )
                            time.sleep(0.015)

                    current_history[-1]["content"] = final_ans
                    yield (
                        current_history,
                        "",
                        token_summary,
                        f"✅ Tool: {selected_tool}",
                        trace_text,
                    )

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        current_history[-1]["content"] = error_msg
        yield current_history, "", "Error", "Failed", [str(e)]


def render_portfolio_agent_tab():
    session_store = gr.State({})

    gr.Markdown(
        """
        # 🤖 Vishnu Thankappan — WebMCP & LangGraph Agent Cockpit
        ### Multi-Provider Architecture (Local Ollama, OpenAI, Hugging Face) with Dynamic Discovery & Token-Efficient LangGraph Routing
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            target_url_input = gr.Textbox(
                value=DEFAULT_TARGET_URL,
                label="🌐 Target Portfolio / MCP URL",
                placeholder="https://www.techiewithbeard.com",
            )
        with gr.Column(scale=1):
            discover_btn = gr.Button("🔍 Discover MCP Tools", variant="primary")

    tool_display = gr.Markdown("Click **Discover MCP Tools** to inspect real-time tool endpoints.")

    with gr.Accordion("⚙️ Model & Provider Configuration", open=False):
        provider_dropdown = gr.Radio(
            choices=PROVIDER_CHOICES,
            value=DEFAULT_PROVIDER,
            label="Provider",
        )
        api_key_input = gr.Textbox(
            type="password",
            label="API Key / Token",
            placeholder="sk-... (Leave blank for 3 free demo queries) or hf_...",
            visible=True,
        )
        base_url_input = gr.Textbox(
            value="https://api.openai.com/v1",
            label="Base Endpoint URL",
        )
        model_name_input = gr.Textbox(
            value="gpt-4o-mini",
            label="Chat Model Name",
        )
        single_turn_cb = gr.Checkbox(
            label="⚡ Single-Turn UI Mode (replace previous turn)",
            value=False,
            info="Keep checked to show only the current query & response.",
        )

    gr.Markdown(
        "💡 **Stateless Architecture**: Zero conversation history is retained or sent to the LLM across turns. "
        "Every query is evaluated independently with pruned tool data to strictly minimize token consumption (<450 tokens/query)."
    )

    chatbot = gr.Chatbot(
        label="Agent Interaction Stream (Stateless: 0 tokens carried over)",
        height=420,
    )

    with gr.Row():
        msg_input = gr.Textbox(
            label="Ask Vishnu's AI Agent",
            placeholder="e.g. What did Vishnu do at AVEVA as Lead Architect?",
            scale=3,
        )
        send_btn = gr.Button("Send Query 🚀", variant="primary", scale=1)
        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)

    with gr.Row():
        active_tool_badge = gr.Markdown("⚡ Tool: None yet")
        token_meter = gr.Markdown("📊 Token Meter: Ready (0 history tokens)")

    with gr.Accordion("🔍 Agent Thought & Tool Trace (Low-Token Execution Log)", open=False):
        trace_output = gr.Markdown("Execution steps will appear here.")

    gr.Examples(
        examples=[
            "What did Vishnu achieve at AVEVA as Lead Frontend Architect?",
            "What are his verified skills in Angular 22 Signals and Nx Monorepos?",
            "Show me his interactive LangGraph AI demos and links.",
            "Is Vishnu available for Lead / Staff roles in Europe?",
        ],
        inputs=msg_input,
    )

    discover_btn.click(
        fn=on_discover_tools,
        inputs=[target_url_input],
        outputs=[tool_display],
    )

    provider_dropdown.change(
        fn=on_provider_change,
        inputs=[provider_dropdown],
        outputs=[api_key_input, base_url_input, model_name_input],
    )

    send_btn.click(
        fn=chat_with_agent,
        inputs=[
            msg_input,
            chatbot,
            target_url_input,
            provider_dropdown,
            api_key_input,
            base_url_input,
            model_name_input,
            session_store,
            single_turn_cb,
        ],
        outputs=[chatbot, msg_input, token_meter, active_tool_badge, trace_output],
    )

    msg_input.submit(
        fn=chat_with_agent,
        inputs=[
            msg_input,
            chatbot,
            target_url_input,
            provider_dropdown,
            api_key_input,
            base_url_input,
            model_name_input,
            session_store,
            single_turn_cb,
        ],
        outputs=[chatbot, msg_input, token_meter, active_tool_badge, trace_output],
    )

    clear_btn.click(
        fn=lambda: ([], "", "📊 Token Meter: 0 tokens (Chat Cleared)", "⚡ Tool: None", "Execution steps will appear here."),
        inputs=[],
        outputs=[chatbot, msg_input, token_meter, active_tool_badge, trace_output],
    )

    return {
        "target_url_input": target_url_input,
        "chatbot": chatbot,
        "msg_input": msg_input,
    }


def create_gradio_ui():
    with gr.Blocks(title="Vishnu Thankappan • WebMCP LangGraph Agent") as demo:
        render_portfolio_agent_tab()
    return demo
