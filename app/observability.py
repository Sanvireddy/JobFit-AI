"""LangSmith tracing helpers, in one place.

LangChain/LangGraph auto-trace themselves; these helpers cover the plain Python
functions those spans would otherwise hide (FAISS retrieval, metadata filtering,
the local Ollama call). Everything here is a no-op when langsmith is absent or
``LANGSMITH_TRACING`` is off, so decorating is free on an untraced run.
"""

import os
from typing import Any, Callable, Optional, Union

try:
    from langsmith import traceable as _ls_traceable
    from langsmith.run_helpers import get_current_run_tree as _get_current_run_tree

    _LANGSMITH_AVAILABLE = True
except Exception:
    _ls_traceable = None

    def _get_current_run_tree():
        return None

    _LANGSMITH_AVAILABLE = False


_TRUE_VALUES = {"1", "true", "yes", "on"}


def tracing_enabled() -> bool:
    """True when langsmith is importable and ``LANGSMITH_TRACING`` is truthy."""
    if not _LANGSMITH_AVAILABLE:
        return False
    return os.environ.get("LANGSMITH_TRACING", "").strip().lower() in _TRUE_VALUES


def traceable(*args: Any, **kwargs: Any) -> Union[Callable, Callable[[Callable], Callable]]:
    """``langsmith.traceable``, or an identity decorator when it is unavailable.

    Supports both ``@traceable`` and ``@traceable(run_type=...)``.
    """
    if _ls_traceable is not None:
        return _ls_traceable(*args, **kwargs)

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def _decorator(func: Callable) -> Callable:
        return func

    return _decorator


def add_trace_metadata(**metadata: Any) -> None:
    """Attach filterable key/values to the current span."""
    run_tree = _get_current_run_tree()
    if run_tree is not None and metadata:
        run_tree.add_metadata(metadata)


def add_trace_tags(*tags: str) -> None:
    """Attach tags to the current span."""
    run_tree = _get_current_run_tree()
    if run_tree is not None and tags:
        run_tree.add_tags([tag for tag in tags if tag])


def log_run_feedback(
    key: str,
    *,
    score: Optional[Union[float, int, bool]] = None,
    value: Optional[Union[str, dict]] = None,
    comment: Optional[str] = None,
) -> None:
    """Record feedback on the current run; best-effort, never raises."""
    run_tree = _get_current_run_tree()
    if run_tree is None:
        return
    try:
        from langsmith import Client

        Client().create_feedback(
            run_id=run_tree.id,
            key=key,
            score=score,
            value=value,
            comment=comment,
        )
    except Exception:
        pass
