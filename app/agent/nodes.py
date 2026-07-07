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
  the COMPLETE new list (``messages`` appends via ``add_messages``;
  ``artifacts``/``applications`` merge per job via their reducers).
- **Fail into ``state["error"]`` instead of raising.** We added an ``error``
  field to the state precisely so a node can signal failure and let a
  conditional edge route to error handling, rather than crashing the whole
  graph run.
- **Pipeline nodes do not write to ``messages``.** They are deterministic
  steps, not model turns, so they touch only the typed field they own.
- **Human-in-the-loop nodes use ``interrupt()``.** ``intake`` and
  ``human_review`` pause the graph and hand a payload to the caller (the CLI),
  which resumes with the human's answer via ``Command(resume=...)``. Both are
  no-ops when there is nothing to ask, so the graph still runs non-interactively
  and without a checkpointer in tests.
"""

from langgraph.types import interrupt

from app.agent.state import AgentState, ApplicationRecord
from app.agent.tools import extract_job_metadata, find_jobs

APPROVE_ANSWERS = {"y", "yes", "apply", "approve", "approved"}


def intake_node(state: AgentState) -> dict:
    """Ask the candidate for their preferences before matching (interactive only).

    Interrupts with a question payload; the caller collects answers and
    resumes with a dict keyed like ``questions``. The answers fill the
    ``CandidateProfile`` preference fields, which feed the compatibility
    filter in ``find_jobs``. A no-op when ``interactive`` is False.
    """
    if not state.get("interactive"):
        return {}

    answers = interrupt({
        "type": "profile_intake",
        "questions": {
            "must_have_skills": (
                "Skills a job must mention, comma-separated (blank for none):"
            ),
            "preferred_locations": (
                "Preferred locations/countries, comma-separated (blank for anywhere):"
            ),
            "open_to_relocation": "Open to relocation? [y/N]:",
            "requires_visa_sponsorship": "Do you need visa sponsorship? [y/N]:",
        },
    })
    if not isinstance(answers, dict):
        answers = {}

    def _csv(key: str) -> list:
        return [part.strip() for part in (answers.get(key) or "").split(",") if part.strip()]

    def _yes(key: str) -> bool:
        return (answers.get(key) or "").strip().lower() in APPROVE_ANSWERS

    candidate = state["candidate"].model_copy(update={
        "must_have_skills": _csv("must_have_skills"),
        "preferred_locations": _csv("preferred_locations"),
        "open_to_relocation": _yes("open_to_relocation"),
        "requires_visa_sponsorship": _yes("requires_visa_sponsorship"),
    })
    return {"candidate": candidate}


def find_jobs_node(state: AgentState) -> dict:
    """Match the candidate's resume to jobs and store the results.

    Reads ``candidate`` and ``top_k`` from state, delegates to ``find_jobs``,
    and writes the resulting matches back. On failure it records the error in
    state (leaving ``matches`` empty) so the graph can branch instead of
    crashing.
    """
    try:
        matches = find_jobs(state["candidate"], top_k=state.get("top_k", 5))
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


def human_review_node(state: AgentState) -> dict:
    """Pause for human approval of every prepared application.

    Interrupts with the list of jobs that have artifacts; the caller resumes
    with per-job decisions (``{job_id: "apply" | "skip"}``). Approved jobs are
    marked ``applied``, the rest ``skipped`` — the model itself can never mark
    anything applied. A no-op when no artifacts were prepared.
    """
    artifacts = state.get("artifacts") or {}
    if not artifacts:
        return {}

    matches = {match.job_id: match for match in state.get("matches") or []}
    decisions = interrupt({
        "type": "application_approval",
        "jobs": [
            {
                "job_id": job_id,
                "title": matches[job_id].title if job_id in matches else "?",
                "company": matches[job_id].company if job_id in matches else "?",
                "has_resume": bool(artifact.tailored_resume),
                "has_cover_letter": bool(artifact.cover_letter),
            }
            for job_id, artifact in artifacts.items()
        ],
    })
    if not isinstance(decisions, dict):
        decisions = {}

    applications = {}
    for job_id in artifacts:
        answer = str(decisions.get(job_id, "")).strip().lower()
        approved = answer in APPROVE_ANSWERS
        applications[job_id] = ApplicationRecord(
            job_id=job_id,
            status="applied" if approved else "skipped",
            notes="approved at human review" if approved else "skipped at human review",
        )
    return {"applications": applications}
