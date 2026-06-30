"""
FAISS vector store: a thin, persistent wrapper around a flat inner-product
index for chunk-level semantic search.

Design notes
------------
- Vectors are L2-normalized before insertion/search so inner product == cosine
  similarity, without needing a separate cosine index type.
- `IndexIDMap2` lets us address vectors by our own integer ids (Chunk.vector_id
  in the SQL DB) instead of FAISS's implicit row order, and supports removal
  by id (used when a document is deleted).
- A single global index is used rather than one index per subject/document.
  Filtering by document_id or subject happens *after* the similarity search,
  by joining the returned vector ids back to the `chunks` table. This trades
  a small amount of search efficiency for much simpler index management —
  a reasonable call at the scale of a personal study tool.
- The index is flat (brute-force) rather than IVF/HNSW: at the scale of a
  few hundred to a few thousand chunks, flat search is both fast enough and
  exact, sidestepping the recall/build-time tradeoffs of approximate indices.
  Noted in the README as a swap-in for production scale.
"""
import json
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss

from app.config import settings

_INDEX_PATH = settings.VECTOR_INDEX_DIR / "chunks.index"
_META_PATH = settings.VECTOR_INDEX_DIR / "meta.json"

_lock = threading.Lock()


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vectors / norms


class VectorStore:
    def __init__(self):
        self.dim = settings.EMBEDDING_DIM
        self._index = None
        self._next_id = 0
        self._load()

    def _load(self):
        if _INDEX_PATH.exists() and _META_PATH.exists():
            self._index = faiss.read_index(str(_INDEX_PATH))
            meta = json.loads(_META_PATH.read_text())
            self._next_id = meta.get("next_id", 0)
        else:
            flat = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIDMap2(flat)
            self._next_id = 0
            self._save()

    def _save(self):
        faiss.write_index(self._index, str(_INDEX_PATH))
        _META_PATH.write_text(json.dumps({"next_id": self._next_id}))

    def add(self, vectors: np.ndarray) -> List[int]:
        """Add vectors, returning the ids assigned to each (in input order)."""
        with _lock:
            if vectors.shape[0] == 0:
                return []
            vectors = _normalize(vectors.astype("float32"))
            ids = np.arange(self._next_id, self._next_id + vectors.shape[0]).astype("int64")
            self._index.add_with_ids(vectors, ids)
            self._next_id += vectors.shape[0]
            self._save()
            return ids.tolist()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return [(vector_id, similarity_score), ...] sorted by relevance desc."""
        if self._index.ntotal == 0:
            return []
        query = _normalize(query_vector.reshape(1, -1).astype("float32"))
        scores, ids = self._index.search(query, min(top_k, self._index.ntotal))
        results = [
            (int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1
        ]
        return results

    def remove(self, ids: List[int]):
        with _lock:
            if not ids:
                return
            self._index.remove_ids(np.array(ids, dtype="int64"))
            self._save()


_store = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
