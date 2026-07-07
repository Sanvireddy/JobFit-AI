"""Agent tools for JobFit-AI.

This module holds two different kinds of callables, and the distinction matters:

1. **Pipeline logic functions** — ``find_jobs`` and ``extract_job_metadata``.
   These are plain typed functions called directly by the deterministic nodes in
   ``app.agent.nodes``. They lazy-import the heavy pipeline (torch/Ollama) inside
   the body so importing this module stays cheap and testable.

2. **Agent action tools** — collected in ``TOOLS``, bound to the Groq model and
   run by the LangGraph ``ToolNode``. They come in three flavors:

   - *Investigation tools* (``analyze_fit``, ``get_job_description``,
     ``check_job_active``, ``research_company``) let the agent gather evidence
     before deciding which jobs to pursue. ``analyze_fit`` is deliberately
     deterministic — it reports facts; the agent supplies the judgment.
   - *Generation tools* (``tailor_resume``, ``write_cover_letter``) produce the
     application documents through a critique-and-revise loop: each draft is
     reviewed by an LLM judge for truthfulness against the original resume and
     regenerated with the critique when rejected (evaluator-optimizer pattern,
     mirroring the validation-retry loop in ``metadata_extractor``).
   - Applying is NOT a tool: prepared materials go through the human-approval
     interrupt in the graph, so the model can never mark anything applied.

   The LLM only supplies a ``job_id`` (or company name); graph state is injected
   via ``InjectedState`` and results are written back to typed state fields via
   a returned ``Command``. Tools return per-job *deltas* — the state reducers
   merge them, which keeps parallel tool calls in one turn safe.
"""

from typing import Annotated, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agent.prompts import (
    COVER_LETTER_SYSTEM_PROMPT,
    DRAFT_REVIEW_SYSTEM_PROMPT,
    TAILOR_RESUME_SYSTEM_PROMPT,
    format_draft_review,
    format_job_and_resume,
)
from app.agent.state import (
    ApplicationRecord,
    CandidateProfile,
    JobMatch,
    TailoredArtifacts,
)
from app.schemas.job_metadata import JobMetadata

# How many times a rejected draft is regenerated with the reviewer's critique
# before we give up and surface the remaining issues to the agent.
MAX_REVISION_ROUNDS = 2


# ---------------------------------------------------------------------------
# Pipeline logic functions (called by nodes, not by the LLM)
# ---------------------------------------------------------------------------


def find_jobs(candidate: CandidateProfile, top_k: int = 5) -> List[JobMatch]:
    """Find the jobs most relevant to a candidate.

    Embeds the resume, searches the FAISS index, applies metadata compatibility
    filtering (experience, degree, language, and — when the profile sets them —
    must-have skills, preferred locations, and visa sponsorship), and returns
    the top matches as ``JobMatch`` records. Metadata already persisted in the
    database rides along, so downstream enrichment only needs to run for jobs
    that lack it.
    """
    # Lazy import: the matcher pulls in sentence-transformers / torch on use.
    from app.matching.matcher import find_matching_jobs_for_resume

    matched = find_matching_jobs_for_resume(
        resume_text=candidate.resume_text,
        top_k=top_k,
        apply_metadata_filtering=True,
        candidate_experience_years=candidate.experience_years,
        must_have_skills=candidate.must_have_skills or None,
        preferred_locations=candidate.preferred_locations or None,
        open_to_relocation=candidate.open_to_relocation,
        requires_visa_sponsorship=candidate.requires_visa_sponsorship,
    )

    return [
        JobMatch(
            job_id=job.job_id,
            title=job.title,
            company=job.company,
            location=job.location,
            application_url=job.application_url,
            similarity_score=job.similarity_score,
            metadata=job.metadata,
            passed_filters=True,
            description=job.description,
        )
        for job in matched
    ]


def extract_job_metadata(job_description: str) -> JobMetadata:
    """Extract structured requirements from a single job description.

    Runs the local LLM extraction and returns a validated ``JobMetadata`` object.
    """
    # Lazy import: metadata_extractor talks to a running Ollama server.
    from app.ingestion.metadata_extractor import extract_metadata

    return extract_metadata(job_description)


# ---------------------------------------------------------------------------
# Shared helpers for the agent tools
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


class DraftReview(BaseModel):
    """Structured verdict from the draft reviewer (LLM-as-judge)."""

    approved: bool = Field(
        description="True only if the draft is truthful against the original "
        "resume and addresses the job's main requirements"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Concrete problems found; empty when approved",
    )


def _review_draft(
    kind: str, draft: str, job_description: str, resume_text: str
) -> DraftReview:
    """Judge a draft for truthfulness and relevance, as structured output.

    Module-level so tests can patch it without a GROQ key.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.agent.llm import get_agent_model

    model = get_agent_model(bind_tools=False).with_structured_output(DraftReview)
    return model.invoke(
        [
            SystemMessage(content=DRAFT_REVIEW_SYSTEM_PROMPT),
            HumanMessage(
                content=format_draft_review(kind, draft, job_description, resume_text)
            ),
        ]
    )


def _generate_reviewed(
    kind: str, system_prompt: str, job_description: str, resume_text: str
) -> tuple:
    """Generate a draft, then critique-and-revise until the reviewer approves.

    Evaluator-optimizer loop: on rejection the reviewer's issues are fed back
    into the generation prompt, up to ``MAX_REVISION_ROUNDS`` times. The last
    draft is returned either way, together with its final review, so the agent
    can see whether it ultimately passed.
    """
    base_prompt = format_job_and_resume(job_description, resume_text)
    draft = _groq_generate(system_prompt, base_prompt)
    review = _review_draft(kind, draft, job_description, resume_text)

    rounds = 0
    while not review.approved and rounds < MAX_REVISION_ROUNDS:
        revision_prompt = (
            base_prompt
            + "\n\nA reviewer rejected your previous draft for these issues:\n- "
            + "\n- ".join(review.issues or ["unspecified issue"])
            + "\n\nPREVIOUS DRAFT:\n"
            + draft
            + "\n\nWrite a corrected draft that resolves every issue. "
            "Stay strictly truthful to the resume."
        )
        draft = _groq_generate(system_prompt, revision_prompt)
        review = _review_draft(kind, draft, job_description, resume_text)
        rounds += 1

    return draft, review


def _review_note(review: DraftReview) -> str:
    if review.approved:
        return "passed the truthfulness review"
    return (
        "review still flags issues after revision: "
        + "; ".join(review.issues or ["unspecified"])
    )


# ---------------------------------------------------------------------------
# Investigation tools (read-only: gather evidence before deciding)
# ---------------------------------------------------------------------------


@tool
def analyze_fit(job_id: str, state: Annotated[dict, InjectedState]) -> str:
    """Get a factual fit report for one shortlisted job.

    Compares the job's extracted requirements and description against the
    candidate's profile: experience gap, degree requirement, work mode, visa
    sponsorship, and coverage of the candidate's must-have skills. The report
    states facts only — judging overall fit is your job.

    Args:
        job_id: The id of a job already present in the shortlist.
    """
    match = _find_match(state, job_id)
    if match is None:
        return f"No shortlisted job with job_id={job_id}."

    candidate: CandidateProfile = state["candidate"]
    lines = [
        f"Fit report for {match.title} at {match.company} (job {job_id}):",
        f"- resume similarity score: {match.similarity_score:.2f}",
    ]

    meta = match.metadata
    experience = meta.experience_requirement if meta else None
    if experience and experience.min_years_experience is not None:
        required = experience.min_years_experience
        have = candidate.experience_years
        if have is None:
            lines.append(
                f"- experience: requires {required}+ years; candidate years unknown"
            )
        elif have >= required:
            lines.append(
                f"- experience: requires {required}+ years; candidate has "
                f"{have} (meets requirement)"
            )
        else:
            lines.append(
                f"- experience: requires {required}+ years; candidate has "
                f"{have} (short by {required - have})"
            )
    else:
        lines.append("- experience: no explicit minimum stated")

    education = meta.higher_education_requirement if meta else None
    if education and education.is_masters_or_phd_required:
        lines.append("- education: explicitly requires a Master's degree or PhD")
    elif education and education.is_masters_or_phd_required is False:
        lines.append("- education: advanced degree not required")
    else:
        lines.append("- education: requirement not stated")

    relocation = meta.relocation_requirement if meta else None
    if relocation:
        lines.append(f"- work mode: {relocation.work_mode}")
        if candidate.requires_visa_sponsorship:
            if relocation.visa_sponsorship_available is True:
                lines.append("- visa: sponsorship available (candidate needs it)")
            elif relocation.visa_sponsorship_available is False:
                lines.append(
                    "- visa: NO sponsorship, but candidate needs it — likely blocker"
                )
            else:
                lines.append("- visa: sponsorship unknown (candidate needs it)")
    else:
        lines.append("- work mode / visa: not stated")

    description = (match.description or "").lower()
    if candidate.must_have_skills:
        present = [s for s in candidate.must_have_skills if s.lower() in description]
        missing = [
            s for s in candidate.must_have_skills if s.lower() not in description
        ]
        if present:
            lines.append(f"- must-have skills mentioned: {', '.join(present)}")
        if missing:
            lines.append(f"- must-have skills NOT mentioned: {', '.join(missing)}")

    if candidate.preferred_locations:
        job_place = " ".join(p for p in (match.location,) if p)
        lines.append(
            f"- location: {job_place or 'unknown'} "
            f"(candidate prefers: {', '.join(candidate.preferred_locations)})"
        )

    return "\n".join(lines)


@tool
def get_job_description(job_id: str, state: Annotated[dict, InjectedState]) -> str:
    """Read the full description text of one shortlisted job.

    Use this when the fit report is ambiguous and you need the posting's own
    words to judge fit.

    Args:
        job_id: The id of a job already present in the shortlist.
    """
    match = _find_match(state, job_id)
    if match is None:
        return f"No shortlisted job with job_id={job_id}."
    if not match.description:
        return f"Job {job_id} has no stored description."
    return (
        f"{match.title} at {match.company} "
        f"({match.location or 'location unknown'}):\n\n{match.description}"
    )


@tool
def check_job_active(
    job_id: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Verify on LinkedIn that a shortlisted job posting is still open.

    Closed postings are removed from the database and marked expired so no
    materials are prepared for them. Requires LinkedIn access; if that is
    unavailable the check reports so and you should assume the job is active.

    Args:
        job_id: The id of the job to verify.
    """
    try:
        # Lazy import: builds a LinkedIn session (cookies via Selenium login).
        from app.ingestion.fetch_jobs import LinkedInJobRetriever

        expired = LinkedInJobRetriever().delete_if_expired(job_id)
    except Exception as exc:
        return Command(update={"messages": [ToolMessage(
            f"Could not verify job {job_id} (LinkedIn access unavailable: {exc}). "
            "Assume the posting is still active.",
            tool_call_id=tool_call_id)]})

    if expired:
        return Command(update={
            "applications": {job_id: ApplicationRecord(
                job_id=job_id, status="expired",
                notes="Posting reported CLOSED by LinkedIn; removed from DB.")},
            "messages": [ToolMessage(
                f"Job {job_id} is CLOSED on LinkedIn. Do not prepare materials "
                "for it.",
                tool_call_id=tool_call_id)],
        })

    return Command(update={"messages": [ToolMessage(
        f"Job {job_id} is still open on LinkedIn.", tool_call_id=tool_call_id)]})


@tool
def research_company(company_name: str) -> str:
    """Look up brief factual background about a company.

    Use one or two returned facts to ground a cover letter. If nothing
    reliable is found, write from the job description alone — never invent
    company facts.

    Args:
        company_name: The hiring company's name as shown in the shortlist.
    """
    import requests

    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": company_name,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return (
            f"Company research unavailable ({exc}). Write the cover letter "
            "from the job description alone."
        )

    abstract = data.get("AbstractText")
    if abstract:
        source = data.get("AbstractSource") or "web"
        return f"About {company_name} (source: {source}): {abstract}"

    topics = [
        topic["Text"]
        for topic in data.get("RelatedTopics", [])
        if isinstance(topic, dict) and topic.get("Text")
    ][:3]
    if topics:
        return f"Notes on {company_name}: " + " | ".join(topics)

    return (
        f"No reliable public summary found for {company_name}. Do not invent "
        "company facts; write from the job description alone."
    )


# ---------------------------------------------------------------------------
# Generation tools (draft -> LLM review -> revise, then write state deltas)
# ---------------------------------------------------------------------------


@tool
def tailor_resume(
    job_id: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Tailor the candidate's resume to one shortlisted job.

    The draft is automatically reviewed for truthfulness against the original
    resume and revised if the reviewer rejects it; the result reports whether
    the final draft passed.

    Args:
        job_id: The id of a job already present in the shortlist.
    """
    match = _find_match(state, job_id)
    if match is None or not match.description:
        return Command(update={"messages": [ToolMessage(
            f"No shortlisted job with a description for job_id={job_id}.",
            tool_call_id=tool_call_id)]})

    draft, review = _generate_reviewed(
        "tailored resume",
        TAILOR_RESUME_SYSTEM_PROMPT,
        match.description,
        state["candidate"].resume_text,
    )

    return Command(update={
        "artifacts": {job_id: TailoredArtifacts(
            job_id=job_id, tailored_resume=draft)},
        "applications": {job_id: ApplicationRecord(
            job_id=job_id, status="tailored")},
        "messages": [ToolMessage(
            f"Tailored resume for {match.title} at {match.company} "
            f"(job {job_id}); {_review_note(review)}.",
            tool_call_id=tool_call_id)],
    })


@tool
def write_cover_letter(
    job_id: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write a cover letter for one shortlisted job.

    The draft is automatically reviewed for truthfulness against the original
    resume and revised if the reviewer rejects it; the result reports whether
    the final draft passed.

    Args:
        job_id: The id of a job already present in the shortlist.
    """
    match = _find_match(state, job_id)
    if match is None or not match.description:
        return Command(update={"messages": [ToolMessage(
            f"No shortlisted job with a description for job_id={job_id}.",
            tool_call_id=tool_call_id)]})

    draft, review = _generate_reviewed(
        "cover letter",
        COVER_LETTER_SYSTEM_PROMPT,
        match.description,
        state["candidate"].resume_text,
    )

    return Command(update={
        "artifacts": {job_id: TailoredArtifacts(
            job_id=job_id, cover_letter=draft)},
        "applications": {job_id: ApplicationRecord(
            job_id=job_id, status="tailored")},
        "messages": [ToolMessage(
            f"Wrote cover letter for {match.title} at {match.company} "
            f"(job {job_id}); {_review_note(review)}.",
            tool_call_id=tool_call_id)],
    })


TOOLS = [
    analyze_fit,
    get_job_description,
    check_job_active,
    research_company,
    tailor_resume,
    write_cover_letter,
]
