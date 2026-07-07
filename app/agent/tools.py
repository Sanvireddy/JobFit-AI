"""Agent tools for JobFit-AI.

Each tool is a thin wrapper over existing pipeline code that takes plain inputs
and returns the typed domain objects defined in :mod:`app.agent.state`. The
heavy pipeline modules (``embeddings`` pulls in sentence-transformers/torch,
``metata_extractor`` talks to Ollama) are **lazy-imported inside the function
bodies** on purpose: importing this module must stay cheap so it can be loaded
and unit-tested without those runtime dependencies present.

Two things live here for each capability:
- a plain typed function (e.g. ``find_jobs``) that holds the logic and is easy
  to call and test directly;
- a LangChain ``StructuredTool`` adapter (e.g. ``find_jobs_tool``) built from it,
  collected in ``TOOLS`` for binding to the model / a LangGraph ``ToolNode``.
"""

from typing import List, Optional

from langchain_core.tools import tool

from app.agent.state import JobMatch
from app.schemas.job_metadata import JobMetadata


def find_jobs(
    resume_text: str,
    top_k: int = 5,
    candidate_experience_years: Optional[int] = 3,
) -> List[JobMatch]:
    """Find the jobs most relevant to a resume.

    Embeds the resume, searches the FAISS index, and applies metadata
    compatibility filtering, returning the top matches as structured
    ``JobMatch`` records.

    Args:
        resume_text: The candidate's resume as plain text.
        top_k: Maximum number of matching jobs to return.
        candidate_experience_years: Years of experience the candidate has; jobs
            requiring more than this are filtered out. Pass ``None`` to skip the
            experience check.
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
    # only approximate. TODO: have the pipeline attach a score to each job (e.g.
    # via a job_id -> score map) and read it here instead.
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
                # These jobs already survived metadata filtering in the pipeline.
                passed_filters=True,
            )
        )
    return matches


def extract_job_metadata(job_description: str) -> JobMetadata:
    """Extract structured requirements from a single job description.

    Runs the local LLM extraction and returns a validated ``JobMetadata``
    object (experience, languages, education, relocation/work mode).

    Args:
        job_description: The raw job description text.
    """
    # Lazy import: metata_extractor talks to a running Ollama server.
    from app.ingestion.metata_extractor import extract_metadata

    return extract_metadata(job_description)


# LLM-facing tool adapters (schemas are inferred from the type hints + docstrings
# above). Bind these to the model or a LangGraph ToolNode.
find_jobs_tool = tool(find_jobs)
extract_job_metadata_tool = tool(extract_job_metadata)

TOOLS = [find_jobs_tool, extract_job_metadata_tool]
