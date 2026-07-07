"""Batch job: extract structured metadata for every unprocessed scraped job.

For each job that has not been processed yet, run the LLM extraction, flatten
the validated ``JobMetadata`` into the ``job_metadata`` table's columns, and
mark the job as processed. Jobs whose extraction fails are reported at the end
and stay unprocessed, so re-running the script retries only the failures.

Run with:  python -m app.ingestion.extract_metadata_batch
"""

import logging
from typing import Optional, Tuple

from app.db.repository import (
    get_unprocessed_jobs,
    insert_jobs_metadata,
    mark_jobs_processed,
)
from app.ingestion.helpers import compute_is_only_english_required
from app.ingestion.metadata_extractor import extract_metadata
from app.schemas.job_metadata import JobMetadata

logger = logging.getLogger(__name__)

BATCH_SIZE = 5


def metadata_to_row(job_id: str, metadata: JobMetadata) -> Tuple:
    """Flatten a validated JobMetadata into a job_metadata table row."""
    experience = metadata.experience_requirement
    education = metadata.higher_education_requirement
    relocation = metadata.relocation_requirement
    only_english, english_evidence = compute_is_only_english_required(
        metadata.language_requirements
    )
    return (
        job_id,
        experience.min_years_experience if experience else None,
        experience.experience_requirement_evidence if experience else None,
        only_english,
        english_evidence,
        education.is_masters_or_phd_required if education else None,
        education.education_requirement_evidence if education else None,
        relocation.visa_sponsorship_available if relocation else None,
        relocation.relocation_assistance_provided if relocation else None,
        relocation.work_mode if relocation else None,
        relocation.relocation_evidence if relocation else None,
    )


def process_all_job_ids(batch_size: int = BATCH_SIZE) -> int:
    """Extract and persist metadata for all unprocessed jobs.

    Returns the number of jobs successfully processed.
    """
    pending = get_unprocessed_jobs()
    if not pending:
        logger.info("No unprocessed jobs found.")
        return 0

    logger.info("Extracting metadata for %d jobs...", len(pending))
    failed: list[Tuple[str, str]] = []
    processed = 0
    batch: list[Tuple] = []

    def flush() -> int:
        nonlocal batch
        if not batch:
            return 0
        try:
            insert_jobs_metadata(batch)
            mark_jobs_processed([row[0] for row in batch])
            logger.info("Persisted metadata for %d jobs", len(batch))
            count = len(batch)
        except Exception as exc:
            failed.extend((row[0], str(exc)) for row in batch)
            logger.exception("Batch insert failed for %d jobs", len(batch))
            count = 0
        batch = []
        return count

    for job_id, description in pending:
        try:
            metadata = extract_metadata(description)
            batch.append(metadata_to_row(job_id, metadata))
        except Exception as exc:
            failed.append((job_id, str(exc)))
            logger.warning("Extraction failed for job_id=%s: %s", job_id, exc)

        if len(batch) >= batch_size:
            processed += flush()

    processed += flush()

    if failed:
        logger.warning("%d jobs failed (re-run to retry):", len(failed))
        for job_id, error in failed:
            logger.warning("  - %s: %s", job_id, error)

    logger.info("Done. Processed %d/%d jobs.", processed, len(pending))
    return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    process_all_job_ids()
