"""Load and persist the FAISS index for job description vectors."""

import logging

import faiss

from app.config import EMBEDDING_DIMENSIONS, FAISS_INDEX_PATH

logger = logging.getLogger(__name__)


def load_or_create_index() -> faiss.Index:
    """Read the index from disk, or start a fresh inner-product flat index.

    Inner product over L2-normalized vectors is cosine similarity.
    """
    if FAISS_INDEX_PATH.exists():
        try:
            return faiss.read_index(str(FAISS_INDEX_PATH))
        except Exception:
            logger.exception(
                "Failed to read FAISS index at %s; creating a new one",
                FAISS_INDEX_PATH,
            )
    return faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)


def save_index(index: faiss.Index) -> None:
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info("Saved FAISS index (%d vectors) to %s", index.ntotal, FAISS_INDEX_PATH)
