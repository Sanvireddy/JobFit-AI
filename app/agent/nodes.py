"""LangGraph nodes for the JobFit-AI agent.

A *node* is a function ``(state: AgentState) -> dict`` where the returned dict is
a **partial** state update that LangGraph merges into the state using each
field's reducer. Nodes are the graph-facing adapter layer: they read what they
need from ``AgentState``, delegate the real work to the pure logic functions in
:mod:`app.agent.tools`, and write results back into typed state fields.

Design decisions (see the module docstring in ``tools.py`` for the tool/node
split):

- **Nodes hold no domain logic.** They only marshal state in and out and call a
  tool. This keeps the matching/extraction logic unit-testable in isolation and
  makes the nodes trivially thin.
- **Return the full field value, not a delta.** ``matches`` has no reducer, so
  LangGraph *overwrites* it. A node that updates matches must therefore return
  the COMPLETE new list (only ``messages`` appends, via its ``add_messages``
  reducer).
- **Fail into ``state["error"]`` instead of raising.** We added an ``error``
  field to the state precisely so a node can signal failure and let a
  conditional edge route to error handling, rather than crashing the whole
  graph run.
- **Nodes do not write to ``messages``.** These are deterministic pipeline
  steps, not model turns, so they stay single-responsibility and touch only the
  typed field they own.
"""

from app.agent.state import AgentState
from app.agent.tools import extract_job_metadata, find_jobs


def find_jobs_node(state: AgentState) -> dict:
    """Match the candidate's resume to jobs and store the results.

    Reads ``candidate`` and ``top_k`` from state, delegates to ``find_jobs``,
    and writes the resulting matches back. On failure it records the error in
    state (leaving ``matches`` empty) so the graph can branch instead of
    crashing.
    """
    candidate = state["candidate"]
    try:
        matches = find_jobs(
            candidate.resume_text,
            top_k=state.get("top_k", 5),
            candidate_experience_years=candidate.experience_years,
        )
    except Exception as exc:
        return {"error": f"find_jobs_node failed: {exc}", "matches": []}

    return {"matches": matches}


def extract_metadata_node(state: AgentState) -> dict:
    """Enrich matches that lack metadata by extracting it from the job text.

    For each match without ``metadata`` (and with a ``description`` to work
    from), run the LLM extraction and attach the resulting ``JobMetadata``.
    Failures on a single job are swallowed so one bad description does not abort
    the batch — that match simply keeps ``metadata=None``.

    Because ``matches`` is overwritten (no reducer), we rebuild and return the
    COMPLETE list, not just the changed items.
    """
    matches = state.get("matches") or []

    enriched = []
    for match in matches:
        if match.metadata is None and match.description:
            try:
                # model_copy(update=...) returns a new JobMatch with metadata set,
                # keeping JobMatch immutable-in-spirit rather than mutating in place.
                match = match.model_copy(
                    update={"metadata": extract_job_metadata(match.description)}
                )
            except Exception:
                # Leave metadata as None on failure; do not abort the batch.
                pass
        enriched.append(match)

    return {"matches": enriched}
