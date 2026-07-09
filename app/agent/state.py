"""State schemas for the JobFit-AI agent.

Two layers live here, kept deliberately separate:

1. Domain schemas (Pydantic ``BaseModel``) — the structured data that flows
   between tools and nodes (the candidate, a matched job, tailored artifacts,
   an application record).
2. The LangGraph runtime state (``AgentState``, a ``TypedDict``) — the mutable
   graph state that nodes read from and write to, with reducers describing how
   updates merge.

This module is intentionally pure: only type/schema definitions, no logic and
no side effects, so tools and the graph can be built on top of a stable
contract. It reuses ``JobMetadata`` from ``app.schemas.job_metadata`` rather
than redefining it.
"""

from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.schemas.job_metadata import JobMetadata


def merge_dicts(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer for dict-valued state fields: merge updates key-by-key.

    Without this, two tool calls in the same superstep (the model routinely
    emits parallel tool calls) would both write the whole dict and LangGraph
    would raise on the concurrent update — or silently drop one. With it,
    tools return only their {job_id: value} delta and updates never collide.
    """
    return {**(left or {}), **(right or {})}


def merge_artifacts(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer for ``artifacts``: merge per job_id AND per field.

    ``tailor_resume`` and ``write_cover_letter`` are routinely called for the
    same job in one model turn, so both deltas land in the same superstep. A
    plain key-level merge would keep only the second artifact (dropping the
    resume or the letter); this merges non-None fields into the existing
    ``TailoredArtifacts`` so both documents survive.
    """
    merged = dict(left or {})
    for job_id, incoming in (right or {}).items():
        current = merged.get(job_id)
        if current is None:
            merged[job_id] = incoming
            continue
        updates = {
            field: value
            for field, value in incoming.model_dump().items()
            if field != "job_id" and value is not None
        }
        merged[job_id] = current.model_copy(update=updates)
    return merged


# ---------------------------------------------------------------------------
# 1. Domain schemas — structured data passed between tools / nodes
# ---------------------------------------------------------------------------


class CandidateProfile(BaseModel):
    """The candidate the agent is finding jobs for"""

    resume_text: str = Field(
        description="Full resume content as plain text"
    )
    experience_years: Optional[int] = Field(
        default=None,
        description=(
            "Years of professional experience the candidate has. Feeds the "
            "experience filter (jobs requiring more than this are dropped). "
            "None skips the experience check"
        ),
    )
    must_have_skills: List[str] = Field(
        default_factory=list,
        description="Skills the candidate wants present in matched jobs",
    )
    preferred_locations: List[str] = Field(
        default_factory=list,
        description="Locations/countries the candidate is targeting",
    )
    open_to_relocation: bool = Field(
        default=False,
        description="Whether the candidate is willing to relocate",
    )
    requires_visa_sponsorship: bool = Field(
        default=False,
        description="Whether the candidate needs visa sponsorship",
    )


class JobMatch(BaseModel):
    """A single job returned by the matching pipeline"""

    job_id: str = Field(description="Stable job identifier (LinkedIn job id)")
    title: str = Field(description="Job title")
    company: str = Field(description="Hiring company name")
    location: Optional[str] = Field(
        default=None, description="Formatted job location"
    )
    application_url: Optional[str] = Field(
        default=None, description="External or easy-apply URL, if available"
    )
    similarity_score: float = Field(
        description="FAISS cosine-similarity score between resume and job"
    )
    metadata: Optional[JobMetadata] = Field(
        default=None,
        description="Structured requirements extracted for this job",
    )
    passed_filters: bool = Field(
        default=False,
        description="Whether the job survived metadata compatibility filtering",
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "Full job description text. Carried on the match so downstream nodes "
            "(metadata extraction, cover-letter tailoring) can reason over the "
            "job's content without re-querying the database"
        ),
    )


class TailoredArtifacts(BaseModel):
    """Per-job documents produced for an application"""

    job_id: str = Field(description="Job this artifact set belongs to")
    tailored_resume: Optional[str] = Field(
        default=None, description="Resume tailored to this job"
    )
    cover_letter: Optional[str] = Field(
        default=None, description="Cover letter tailored for this job"
    )


class ScreeningDecision(BaseModel):
    """The screener agent's verdict on one shortlisted job.

    This is the typed handoff between the two agents: the screener records one
    decision per job; the preparer only ever sees jobs with ``pursue=True``
    (rendered by ``format_handoff``) and the reasons behind them.
    """

    job_id: str = Field(description="Job this decision is about")
    pursue: bool = Field(
        description="True to hand the job to the preparer agent for materials"
    )
    reason: str = Field(
        description="One or two sentences justifying the decision"
    )


ApplicationStatus = Literal[
    "shortlisted",
    "tailored",
    "awaiting_approval",
    "applied",
    "skipped",
    "expired",
]


class ApplicationRecord(BaseModel):
    """Tracks where a given job is in the application workflow"""

    job_id: str = Field(description="Job this record tracks")
    status: ApplicationStatus = Field(
        default="shortlisted",
        description="Current stage of the application for this job",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Free-text notes (e.g. why a job was skipped)",
    )


# ---------------------------------------------------------------------------
# 2. LangGraph runtime state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Mutable state threaded through the agent graph.

    Reducer note: ``messages`` uses ``add_messages`` so each agent/tool turn
    *appends* to the conversation rather than overwriting it — this is what
    makes the tool-calling loop work. ``artifacts`` and ``applications`` use
    ``merge_dicts`` so tools return per-job deltas that merge safely even when
    the model emits several tool calls in one turn. The remaining fields use
    the default "overwrite on update" behavior, which is appropriate since a
    node computes the full new value (e.g. the complete ``matches`` list).
    """

    # Conversation history between the model and the tools.
    messages: Annotated[list, add_messages]

    # Who we are matching, and how many results to aim for.
    candidate: CandidateProfile
    top_k: int

    # When True, the intake node interrupts to ask the candidate for their
    # preferences (skills, locations, visa) before matching runs.
    interactive: bool

    # Working data produced as the graph runs. The dicts merge per-key (see
    # merge_dicts) so parallel tool calls can update different jobs safely.
    matches: List[JobMatch]
    screening: Annotated[Dict[str, ScreeningDecision], merge_dicts]  # keyed by job_id
    artifacts: Annotated[Dict[str, TailoredArtifacts], merge_artifacts]
    applications: Annotated[Dict[str, ApplicationRecord], merge_dicts]

    # The screener agent's closing summary, stashed by the handoff node before
    # the message channel is cleared for the preparer's isolated context.
    screener_summary: Optional[str]

    # Populated if a node fails, so the graph can route to error handling.
    error: Optional[str]
