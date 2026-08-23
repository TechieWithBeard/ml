from time import perf_counter
from typing import Any


def log_llm_metrics(
    node_name: str,
    response: Any,
    elapsed: float,
) -> None:

    metadata = getattr(
        response,
        "response_metadata",
        {},
    ) or {}

    def seconds(value: Any) -> float | None:
        if value is None:
            return None
        return value / 1_000_000_000

    total = seconds(metadata.get("total_duration"))
    load = seconds(metadata.get("load_duration"))
    prompt_eval = seconds(
        metadata.get("prompt_eval_duration")
    )
    eval_duration = seconds(
        metadata.get("eval_duration")
    )

    output_tokens = metadata.get("eval_count")

    tokens_per_second = None

    if eval_duration and output_tokens:
        tokens_per_second = (
            output_tokens / eval_duration
        )

    print(
        f"""
========== LLM PERFORMANCE ==========
Node:                 {node_name}
Model:                {metadata.get("model")}

Wall time:            {elapsed:.3f}s
Ollama total:         {total}
Model load:           {load}
Prompt evaluation:    {prompt_eval}
Generation:           {eval_duration}

Input tokens:         {metadata.get("prompt_eval_count")}
Output tokens:        {output_tokens}
Generation speed:     {tokens_per_second}

======================================
"""
    )