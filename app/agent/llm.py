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

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

from app.agent.tools import TOOLS
from app.config import GROQ_DEFAULT_MODEL as DEFAULT_MODEL

# On Groq's free/on-demand tier the per-minute token budget (TPM) is small, and
# the agent fires calls in a tight ReAct/evaluator loop — enough to exhaust the
# window in seconds and get a 429. Two defenses, both env-tunable:
#   - retry: the Groq SDK honors the API's Retry-After on 429, so a higher retry
#     budget rides out transient throttles instead of crashing the run.
#   - pace: a process-wide limiter spaces requests out so the rolling TPM window
#     can recover between calls.
GROQ_MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "8"))
# Requests per second, shared across every model built here. Default ~0.1 rps
# (one call every ~10s) keeps a ~2.5k-token request comfortably under a 12k TPM
# cap; raise it on paid tiers via GROQ_REQUESTS_PER_SECOND.
_RPS = float(os.environ.get("GROQ_REQUESTS_PER_SECOND", "0.1"))
_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=_RPS,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)


def get_agent_model(
    model: Optional[str] = None,
    temperature: float = 0.0,
    bind_tools: bool = True,
    tools: Optional[list] = None,
) -> ChatGroq:
    """Return a configured Groq chat model, optionally bound with agent tools.

    Args:
        model: Explicit model id. Falls back to ``GROQ_MODEL`` then
            ``DEFAULT_MODEL``.
        temperature: Sampling temperature; 0.0 for deterministic tool routing
            and structured extraction.
        bind_tools: When True (default), bind tools so the model can emit
            tool calls for a LangGraph ``ToolNode`` to execute.
        tools: Specific tool roster to bind (e.g. ``SCREENER_TOOLS``).
            Defaults to the full ``TOOLS`` union when ``bind_tools`` is True.

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
        max_retries=GROQ_MAX_RETRIES,
        rate_limiter=_RATE_LIMITER,
    )

    if tools is not None:
        return llm.bind_tools(tools)
    return llm.bind_tools(TOOLS) if bind_tools else llm
