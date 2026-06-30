"""
Local text embeddings via sentence-transformers.

Deliberately local and free (no API key, no per-call cost) rather than
calling an embeddings API: embedding generation happens far more often than
LLM generation in a RAG pipeline (every chunk, every query), so keeping it
local keeps the project runnable end-to-end on a student budget. The LLM
call itself (the expensive, reasoning-heavy part) is still delegated to a
hosted model — see llm_client.py.
"""
from functools import lru_cache
from typing import List
import numpy as np

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    # Imported lazily so the rest of the app can be imported/tested without
    # pulling in torch if embeddings aren't needed yet.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts. Returns a (n, EMBEDDING_DIM) float32 array."""
    if not texts:
        return np.zeros((0, settings.EMBEDDING_DIM), dtype="float32")
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype("float32")


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns a (EMBEDDING_DIM,) float32 array."""
    return embed_texts([text])[0]
