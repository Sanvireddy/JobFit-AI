"""Query-time resume-to-job matching.

Pipeline: embed resume -> FAISS nearest-neighbor search -> hydrate job rows
from SQLite -> attach similarity scores -> metadata compatibility filtering ->
top-k ``MatchedJob`` results.

Design notes:
- Failures raise; callers decide how to handle them. (The old version returned
  a ``{"success": ..., "error": ...}`` dict that every caller had to unpack.)
- Each job carries its own similarity score. Previously scores were returned
  as a separate list aligned by search rank, which silently misaligned once
  filtering reordered or dropped jobs.
- ``filter_jobs_by_metadata`` and ``metadata_from_row`` are pure functions so
  the filtering rules are unit-testable without a database or model.
"""

import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.db import repository
from app.schemas.job_metadata import (
    ExperienceRequirement,
    HigherEducationRequirement,
    JobMetadata,
    RelocationRequirement,
)

logger = logging.getLogger(__name__)


class MatchedJob(BaseModel):
    """One job returned by the matcher, with its similarity score attached."""

    job_id: str
    title: str
    company: str
    location: Optional[str] = None
    application_url: Optional[str] = None
    description: Optional[str] = None
    similarity_score: float = Field(
        description="Cosine similarity between the resume and this job"
    )
    metadata: Optional[JobMetadata] = Field(
        default=None,
        description="Requirements previously extracted for this job, if stored",
    )


def metadata_from_row(row: Optional[Dict]) -> Optional[JobMetadata]:
    """Rebuild a partial ``JobMetadata`` from a flattened job_metadata DB row.

    The DB stores a flattened projection of the extraction output, so this is
    lossy (e.g. the per-language breakdown is not recoverable). Sub-models are
    only built when their required evidence fields are present; if nothing can
    be rebuilt, returns None and the caller may re-extract from the raw text.
    """
    if not row:
        return None

    experience = None
    if row.get("experience_requirement_text"):
        experience = ExperienceRequirement(
            min_years_experience=row.get("min_experience_years"),
            experience_requirement_evidence=row["experience_requirement_text"],
        )

    education = None
    if row.get("requires_advanced_degree") is not None or row.get(
        "education_requirement_text"
    ):
        education = HigherEducationRequirement(
            is_masters_or_phd_required=row.get("requires_advanced_degree"),
            education_requirement_evidence=row.get("education_requirement_text"),
        )

    relocation = None
    if row.get("relocation_evidence"):
        relocation = RelocationRequirement(
            visa_sponsorship_available=row.get("visa_sponsorship_available"),
            relocation_assistance_provided=row.get("relocation_assistance_provided"),
            work_mode=row.get("work_mode") or "unknown",
            relocation_evidence=row["relocation_evidence"],
        )

    if experience is None and education is None and relocation is None:
        return None

    return JobMetadata(
        experience_requirement=experience,
        higher_education_requirement=education,
        relocation_requirement=relocation,
    )


def filter_jobs_by_metadata(
    jobs: List[Dict],
    job_metadata_map: Dict[str, Dict],
    candidate_experience_years: Optional[int] = None,
    must_have_skills: Optional[List[str]] = None,
    preferred_locations: Optional[List[str]] = None,
    open_to_relocation: bool = False,
    requires_visa_sponsorship: bool = False,
) -> List[Dict]:
    """Keep only jobs the candidate is plausibly compatible with.

    A job is kept when all of the following hold:
    - Its required minimum experience is <= the candidate's experience (the
      check is skipped when ``candidate_experience_years`` is None).
    - It does not explicitly require an advanced degree (Master's/PhD).
    - It does not explicitly require a non-English language. Jobs whose
      language requirement is unknown are kept, since "unknown" should not
      silently drop otherwise-relevant matches.
    - Every ``must_have_skills`` entry appears in the job description
      (case-insensitive substring; skipped when the list is empty).
    - The job is in one of ``preferred_locations`` — or is remote, or the
      candidate is open to relocation (skipped when the list is empty).
    - When the candidate needs visa sponsorship, jobs that explicitly say no
      sponsorship are dropped; unknown (None) is kept, mirroring the language
      rule: unknowns should not silently drop otherwise-relevant matches.
    """
    filtered = []
    for job in jobs:
        metadata = job_metadata_map.get(job.get("job_id"), {})

        # Experience: be defensive about None / non-numeric DB values.
        raw_exp = metadata.get("min_experience_years")
        try:
            required_years = int(raw_exp) if raw_exp is not None else 0
        except (TypeError, ValueError):
            required_years = 0
        if (
            candidate_experience_years is not None
            and required_years > candidate_experience_years
        ):
            continue

        if metadata.get("requires_advanced_degree"):
            continue

        # requires_only_english == False means a non-English language is
        # mandatory; True (English-only) and None (unknown) are both kept.
        if metadata.get("requires_only_english") is False:
            continue

        if must_have_skills:
            description = (job.get("description") or "").lower()
            if not all(skill.lower() in description for skill in must_have_skills):
                continue

        if preferred_locations and not open_to_relocation:
            job_place = " ".join(
                part for part in (job.get("location"), job.get("country")) if part
            ).lower()
            is_remote = metadata.get("work_mode") == "remote"
            in_preferred = any(
                loc.lower() in job_place for loc in preferred_locations if loc.strip()
            )
            if not (is_remote or in_preferred):
                continue

        if (
            requires_visa_sponsorship
            and metadata.get("visa_sponsorship_available") is False
        ):
            continue

        filtered.append(job)
    return filtered


def find_matching_jobs_for_resume(
    resume_text: str,
    top_k: int = 10,
    apply_metadata_filtering: bool = True,
    candidate_experience_years: Optional[int] = None,
    must_have_skills: Optional[List[str]] = None,
    preferred_locations: Optional[List[str]] = None,
    open_to_relocation: bool = False,
    requires_visa_sponsorship: bool = False,
) -> List[MatchedJob]:
    """Return the top-k jobs most similar to a resume, best match first.

    Searches a wider candidate pool (2x top_k) so that metadata filtering can
    drop incompatible jobs without starving the final result.
    """
    # Imported here so this module stays importable without torch installed.
    from app.embeddings.encoder import embed_texts
    from app.embeddings.index_store import load_or_create_index

    index = load_or_create_index()
    if index.ntotal == 0:
        logger.warning("FAISS index is empty; run app.ingestion.build_index first.")
        return []

    resume_vector = embed_texts([resume_text])
    scores, indices = index.search(resume_vector, top_k * 2)
    score_by_faiss_index = {
        int(idx): float(score)
        for idx, score in zip(indices[0], scores[0])
        if idx != -1  # FAISS pads with -1 when fewer results exist
    }

    jobs = repository.get_job_details_by_faiss_indices(
        list(score_by_faiss_index.keys())
    )
    metadata_map = repository.get_job_metadata_by_job_ids(
        [job["job_id"] for job in jobs]
    )

    if apply_metadata_filtering:
        jobs = filter_jobs_by_metadata(
            jobs,
            metadata_map,
            candidate_experience_years=candidate_experience_years,
            must_have_skills=must_have_skills,
            preferred_locations=preferred_locations,
            open_to_relocation=open_to_relocation,
            requires_visa_sponsorship=requires_visa_sponsorship,
        )

    jobs.sort(
        key=lambda job: score_by_faiss_index.get(job["faiss_index"], 0.0),
        reverse=True,
    )

    return [
        MatchedJob(
            job_id=str(job["job_id"]),
            title=job.get("title") or "",
            company=job.get("company") or "",
            location=job.get("location"),
            application_url=job.get("application_url"),
            description=job.get("description"),
            similarity_score=score_by_faiss_index.get(job["faiss_index"], 0.0),
            metadata=metadata_from_row(metadata_map.get(job["job_id"])),
        )
        for job in jobs[:top_k]
    ]
