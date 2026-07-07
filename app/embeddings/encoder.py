"""Text embedding via sentence-transformers.

The model is loaded lazily and cached. Loading it at import time (as the old
``embeddings.py`` did) pulled torch into every process that merely imported
the module — including test runs and the agent's import graph.
"""

from functools import lru_cache

import numpy as np

from app.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list) -> np.ndarray:
    """Encode texts into L2-normalized float32 vectors (rows)."""
    embeddings = get_embedding_model().encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)
