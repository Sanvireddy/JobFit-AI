"""Agent tools for JobFit-AI.

This module holds two different kinds of callables, and the distinction matters:

1. **Pipeline logic functions** — ``find_jobs`` and ``extract_job_metadata``.
   These are plain typed functions called directly by the deterministic nodes in
   ``app.agent.nodes``. They lazy-import the heavy pipeline (torch/Ollama) inside
   the body so importing this module stays cheap and testable.

2. **Agent action tools** — ``tailor_resume``, ``write_cover_letter``,
   ``mark_applied`` (collected in ``TOOLS``). These are what the Groq agent
   actually calls to do work. The LLM only supplies a ``job_id``; the graph state
   is injected via ``InjectedState`` and results are written back to typed state
   fields via a returned ``Command``. These are the tools bound to the model and
   run by the LangGraph ``ToolNode``.
"""

from typing import Annotated, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.agent.state import ApplicationRecord, JobMatch, TailoredArtifacts
from app.schemas.job_metadata import JobMetadata


# ---------------------------------------------------------------------------
# Pipeline logic functions (called by nodes, not by the LLM)
# ---------------------------------------------------------------------------


def find_jobs(
    resume_text: str,
    top_k: int = 5,
    candidate_experience_years: Optional[int] = 3,
) -> List[JobMatch]:
    """Find the jobs most relevant to a resume.

    Embeds the resume, searches the FAISS index, applies metadata compatibility
    filtering, and returns the top matches as ``JobMatch`` records.
    """
    # Lazy import: embeddings pulls in sentence-transformers / torch and builds
    # the embedding model at import time, so keep it out of module import.
    from app.ingestion.embeddings import find_matching_jobs_for_resume

    result = find_matching_jobs_for_resume(
        resume_text=resume_text,
        top_k=top_k,
        apply_metadata_filtering=True,
        candidate_experience_years=candidate_experience_years,
    )

    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Job matching failed.")

    # NOTE: faiss_scores are aligned to search rank, but the underlying pipeline
    # reorders/filters jobs before returning them, so this positional mapping is
    # only approximate. TODO: have the pipeline attach a score to each job.
    scores = result.get("faiss_scores") or []

    matches: List[JobMatch] = []
    for i, job in enumerate(result.get("similar_jobs", [])):
        matches.append(
            JobMatch(
                job_id=str(job["job_id"]),
                title=job.get("title") or "",
                company=job.get("company") or "",
                location=job.get("location"),
                application_url=job.get("application_url"),
                similarity_score=float(scores[i]) if i < len(scores) else 0.0,
                passed_filters=True,
                description=job.get("description"),
            )
        )
    return matches


def extract_job_metadata(job_description: str) -> JobMetadata:
    """Extract structured requirements from a single job description.

    Runs the local LLM extraction and returns a validated ``JobMetadata`` object.
    """
    # Lazy import: metata_extractor talks to a running Ollama server.
    from app.ingestion.metata_extractor import extract_metadata

    return extract_metadata(job_description)


# ---------------------------------------------------------------------------
# Agent action tools (called by the LLM; read state, write state)
# ---------------------------------------------------------------------------


def _find_match(state: dict, job_id: str) -> Optional[JobMatch]:
    for match in state.get("matches") or []:
        if match.job_id == job_id:
            return match
    return None


def _groq_generate(system_prompt: str, user_prompt: str) -> str:
    """Single-shot Groq generation used by the tailoring tools.

    Kept as a module-level function so tests can patch it without a GROQ key.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.agent.llm import get_agent_model

    model = get_agent_model(bind_tools=False)
    reply = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return reply.content


@tool
def tailor_resume(
    job_id: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Tailor the candidate's resume to one shortlisted job.

    Args:
        job_id: The id of a job already present in the shortlist.
    """
    match = _find_match(state, job_id)
    if match is None or not match.description:
        return Command(update={"messages": [ToolMessage(
            f"No shortlisted job with a description for job_id={job_id}.",
            tool_call_id=tool_call_id)]})

    tailored = _groq_generate(
        "You tailor resumes to a specific job. Rewrite the resume to emphasize "
        "experience relevant to the job. Stay strictly truthful; never invent "
        "experience the candidate does not have.",
        f"JOB DESCRIPTION:\n{match.description}\n\nRESUME:\n"
        f"{state['candidate'].resume_text}",
    )

    artifacts = dict(state.get("artifacts") or {})
    existing = artifacts.get(job_id) or TailoredArtifacts(job_id=job_id)
    artifacts[job_id] = existing.model_copy(update={"tailored_resume": tailored})

    return Command(update={
        "artifacts": artifacts,
        "messages": [ToolMessage(
            f"Tailored resume for {match.title} at {match.company} (job {job_id}).",
            tool_call_id=tool_call_id)],
    })


@tool
def write_cover_letter(
    job_id: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write a cover letter for one shortlisted job.

    Args:
        job_id: The id of a job already present in the shortlist.
    """
    match = _find_match(state, job_id)
    if match is None or not match.description:
        return Command(update={"messages": [ToolMessage(
            f"No shortlisted job with a description for job_id={job_id}.",
            tool_call_id=tool_call_id)]})

    letter = _groq_generate(
        "You write concise, specific cover letters grounded in the candidate's "
        "real experience. Do not invent facts.",
        f"JOB DESCRIPTION:\n{match.description}\n\nRESUME:\n"
        f"{state['candidate'].resume_text}",
    )

    artifacts = dict(state.get("artifacts") or {})
    existing = artifacts.get(job_id) or TailoredArtifacts(job_id=job_id)
    artifacts[job_id] = existing.model_copy(update={"cover_letter": letter})

    return Command(update={
        "artifacts": artifacts,
        "messages": [ToolMessage(
            f"Wrote cover letter for {match.title} at {match.company} (job {job_id}).",
            tool_call_id=tool_call_id)],
    })


@tool
def mark_applied(
    job_id: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Record that the candidate has applied to a job (status tracking only).

    This updates application status in state; it does NOT submit anything
    externally. Real submission should sit behind a human-approval gate.

    Args:
        job_id: The id of the job to mark as applied.
    """
    applications = dict(state.get("applications") or {})
    existing = applications.get(job_id) or ApplicationRecord(job_id=job_id)
    applications[job_id] = existing.model_copy(update={"status": "applied"})

    return Command(update={
        "applications": applications,
        "messages": [ToolMessage(
            f"Marked job {job_id} as applied.", tool_call_id=tool_call_id)],
    })


TOOLS = [tailor_resume, write_cover_letter, mark_applied]
