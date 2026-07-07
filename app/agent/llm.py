"""LLM provider for the JobFit-AI agent (Groq).

The agent's reasoning model runs on Groq via ``langchain-groq``. It is built in a
factory function rather than at import time so that importing this module never
requires ``GROQ_API_KEY`` — matching the cheap, side-effect-free import
convention used across ``app.agent`` (tools lazy-import torch/Ollama; state is
pure schema).

Configuration (both read from the environment, never hard-coded):
- ``GROQ_API_KEY`` — required at call time; a clear error is raised if missing.
- ``GROQ_MODEL``   — optional override of the default model.

The default is a tool-calling-capable model, since the agent node binds tools to
it. Groq also hosts smaller/faster options (e.g. ``llama-3.1-8b-instant``) and
other tool-capable models; override via ``GROQ_MODEL`` without touching code.
"""

import os
from typing import Optional

from langchain_groq import ChatGroq

from app.agent.tools import TOOLS

# Tool-calling-capable default. Override with the GROQ_MODEL env var.
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_agent_model(
    model: Optional[str] = None,
    temperature: float = 0.0,
    bind_tools: bool = True,
) -> ChatGroq:
    """Return a configured Groq chat model, optionally bound with the agent tools.

    Args:
        model: Explicit model id. Falls back to ``GROQ_MODEL`` then
            ``DEFAULT_MODEL``.
        temperature: Sampling temperature; 0.0 for deterministic tool routing
            and structured extraction.
        bind_tools: When True (default), bind ``TOOLS`` so the model can emit
            tool calls for a LangGraph ``ToolNode`` to execute.

    Raises:
        RuntimeError: if ``GROQ_API_KEY`` is not set.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export your Groq key before running the "
            "agent, e.g. `export GROQ_API_KEY=gsk_...`."
        )

    # ChatGroq reads GROQ_API_KEY from the environment itself; we validate above
    # only to fail with a friendly message instead of a cryptic auth error.
    llm = ChatGroq(
        model=model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        temperature=temperature,
    )

    return llm.bind_tools(TOOLS) if bind_tools else llm
