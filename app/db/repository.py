"""SQLite persistence layer for JobFit-AI.

Every function opens a short-lived connection against ``config.DB_PATH`` so
callers never manage connections or cursors themselves. Schema creation is
centralized in :func:`init_db` instead of being scattered across insert
functions, so the code's view of the schema lives in exactly one place.
"""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Dict, Iterator, List, Sequence, Tuple

from app.config import DB_PATH

logger = logging.getLogger(__name__)


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Yield a connection that commits on success and always closes."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:  # transaction scope: commit on success, rollback on error
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they do not exist yet."""
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS job_processing_status (
            job_id TEXT UNIQUE,
            is_scraped BOOLEAN DEFAULT 0 NOT NULL CHECK (is_scraped IN (0, 1))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scraped_jobs (
            job_id TEXT UNIQUE PRIMARY KEY,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            description TEXT NOT NULL,
            location TEXT,
            posted_date TEXT,
            application_url TEXT,
            country TEXT,
            is_processed BOOLEAN DEFAULT 0,
            faiss_index INTEGER
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS job_metadata (
            job_id TEXT UNIQUE PRIMARY KEY,
            min_experience_years INTEGER,
            experience_requirement_text TEXT,
            requires_only_english BOOLEAN,
            language_requirement_text TEXT,
            requires_advanced_degree BOOLEAN,
            education_requirement_text TEXT,
            visa_sponsorship_available BOOLEAN,
            relocation_assistance_provided BOOLEAN,
            work_mode TEXT,
            relocation_evidence TEXT,
            FOREIGN KEY (job_id) REFERENCES scraped_jobs(job_id)
        )
        """,
    ]
    with _connect() as conn:
        for statement in ddl:
            conn.execute(statement)


# ---------------------------------------------------------------------------
# Ingestion: job ids and scraped details
# ---------------------------------------------------------------------------


def insert_fetched_job_ids(job_ids: Sequence[str]) -> None:
    """Record newly discovered job ids (idempotent)."""
    init_db()
    with _connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO job_processing_status (job_id) VALUES (?)",
            [(job_id,) for job_id in job_ids],
        )


def get_unscraped_job_ids() -> List[str]:
    """Job ids we discovered but have not fetched full details for yet."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT job_id FROM job_processing_status WHERE is_scraped = 0"
        ).fetchall()
    return [row["job_id"] for row in rows]


def insert_job_details(job_details: Sequence[Tuple]) -> None:
    """Store fully scraped job records and mark their ids as scraped.

    Each tuple: (job_id, title, company, description, location, posted_date,
    application_url, country).
    """
    if not job_details:
        return
    init_db()
    job_ids = [detail[0] for detail in job_details]
    placeholders = ",".join("?" * len(job_ids))
    with _connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO scraped_jobs "
            "(job_id, title, company, description, location, posted_date, "
            "application_url, country) VALUES (?,?,?,?,?,?,?,?)",
            job_details,
        )
        conn.execute(
            f"UPDATE job_processing_status SET is_scraped = 1 "
            f"WHERE job_id IN ({placeholders})",
            job_ids,
        )


def delete_jobs_with_excluded_titles(
    patterns: Sequence[str] = ("%intern%", "%part%"),
) -> int:
    """Drop jobs whose titles match unwanted patterns (e.g. internships).

    Kept as an explicit, separately-called cleanup step — previously this ran
    hidden inside the insert path, which made data silently disappear.
    Returns the number of deleted rows.
    """
    clause = " OR ".join("LOWER(title) LIKE ?" for _ in patterns)
    with _connect() as conn:
        cursor = conn.execute(
            f"DELETE FROM scraped_jobs WHERE {clause}", list(patterns)
        )
    logger.info("Deleted %d jobs with excluded titles", cursor.rowcount)
    return cursor.rowcount


def delete_job(job_id: str) -> None:
    """Remove a single job (e.g. because the posting expired)."""
    with _connect() as conn:
        conn.execute("DELETE FROM scraped_jobs WHERE job_id = ?", (job_id,))


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def get_unprocessed_jobs() -> List[Tuple[str, str]]:
    """(job_id, description) pairs that still need metadata extraction."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT job_id, description FROM scraped_jobs WHERE is_processed = 0"
        ).fetchall()
    return [(row["job_id"], row["description"]) for row in rows]


def insert_jobs_metadata(metadata_rows: Sequence[Tuple]) -> None:
    """Upsert extracted metadata rows (see init_db for column order)."""
    init_db()
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO job_metadata (
                job_id,
                min_experience_years,
                experience_requirement_text,
                requires_only_english,
                language_requirement_text,
                requires_advanced_degree,
                education_requirement_text,
                visa_sponsorship_available,
                relocation_assistance_provided,
                work_mode,
                relocation_evidence
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            metadata_rows,
        )


def mark_jobs_processed(job_ids: Sequence[str]) -> None:
    if not job_ids:
        return
    placeholders = ",".join("?" * len(job_ids))
    with _connect() as conn:
        conn.execute(
            f"UPDATE scraped_jobs SET is_processed = 1 "
            f"WHERE job_id IN ({placeholders})",
            list(job_ids),
        )


# ---------------------------------------------------------------------------
# Embeddings / vector index bookkeeping
# ---------------------------------------------------------------------------


def get_jobs_without_embeddings() -> List[Tuple[str, str]]:
    """(job_id, description) pairs not yet added to the FAISS index."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT job_id, description FROM scraped_jobs WHERE faiss_index IS NULL"
        ).fetchall()
    return [(row["job_id"], row["description"]) for row in rows]


def set_faiss_index(job_id: str, faiss_index: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE scraped_jobs SET faiss_index = ? WHERE job_id = ?",
            (faiss_index, job_id),
        )


# ---------------------------------------------------------------------------
# Matching-time reads
# ---------------------------------------------------------------------------


def get_job_details_by_faiss_indices(faiss_indices: Sequence[int]) -> List[Dict]:
    """Hydrate full job rows for FAISS search hits.

    ``faiss_index`` is included in the result so callers can map each job back
    to its similarity score.
    """
    if not faiss_indices:
        return []
    placeholders = ",".join("?" * len(faiss_indices))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT job_id, title, company, description, location, posted_date, "
            "application_url, country, faiss_index "
            f"FROM scraped_jobs WHERE faiss_index IN ({placeholders})",
            list(faiss_indices),
        ).fetchall()
    return [dict(row) for row in rows]


def get_job_metadata_by_job_ids(job_ids: Sequence[str]) -> Dict[str, Dict]:
    """Extracted metadata keyed by job_id (missing ids are simply absent)."""
    if not job_ids:
        return {}
    placeholders = ",".join("?" * len(job_ids))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT job_id, min_experience_years, experience_requirement_text, "
            "requires_only_english, requires_advanced_degree, "
            "education_requirement_text, visa_sponsorship_available, "
            "relocation_assistance_provided, work_mode, relocation_evidence "
            f"FROM job_metadata WHERE job_id IN ({placeholders})",
            list(job_ids),
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
