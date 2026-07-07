"""Batch job: embed new job descriptions and add them to the FAISS index.

For every scraped job that has no ``faiss_index`` yet, encode its description,
append the vector to the index, and record the vector's position back in
SQLite so matching-time searches can map hits to job rows.

Run with:  python -m app.ingestion.build_index
"""

import logging

from app.db import repository
from app.embeddings.encoder import embed_texts
from app.embeddings.index_store import load_or_create_index, save_index

logger = logging.getLogger(__name__)


def index_new_jobs() -> int:
    """Embed and index all jobs missing from the FAISS index.

    Returns the number of jobs indexed.
    """
    pending = repository.get_jobs_without_embeddings()
    if not pending:
        logger.info("No new jobs to index.")
        return 0

    index = load_or_create_index()
    indexed = 0

    for job_id, description in pending:
        if not description:
            logger.warning("Skipping job_id=%s: empty description", job_id)
            continue
        try:
            vector = embed_texts([description])
            index.add(vector)
            position = index.ntotal - 1
            repository.set_faiss_index(job_id, position)
            indexed += 1
            logger.info("Indexed job_id=%s at position %d", job_id, position)
        except Exception:
            logger.exception("Failed to index job_id=%s", job_id)

    save_index(index)
    return indexed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    index_new_jobs()
