from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

try:
    import faiss  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    faiss = None


class VectorStore:
    """Lightweight vector index with optional FAISS backend.

    - ``dim`` sets the fixed embedding dimensionality.
    - ``use_faiss=True`` attempts to initialize FAISS and NumPy; on any import failure, a pure-Python
      L2 search is used instead.
    - ``search`` always returns ``(id, distance)`` pairs ordered by ascending distance, regardless of
      backend availability.
    """

    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        self.dim = dim
        self._id_to_vector: Dict[str, List[float]] = {}
        self._use_faiss = bool(use_faiss and faiss is not None)
        self._faiss_index = faiss.IndexFlatL2(self.dim) if self._use_faiss else None  # type: ignore[attr-defined]
        self._faiss_ids: List[str] = []

    def _ensure_dim(self, vector: Sequence[float]) -> List[float]:
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {len(vector)}")
        return [float(x) for x in vector]

    def _rebuild_faiss(self) -> None:
        if not self._use_faiss:
            return
        try:
            import numpy as np  # type: ignore
        except Exception:
            self._use_faiss = False
            self._faiss_index = None
            self._faiss_ids = []
            return
        self._faiss_index = faiss.IndexFlatL2(self.dim) if faiss is not None else None  # type: ignore[attr-defined]
        self._faiss_ids = []
        if self._faiss_index is None:
            return
        if not self._id_to_vector:
            return
        ids = list(self._id_to_vector.keys())
        vectors = [self._id_to_vector[i] for i in ids]
        arr = np.asarray(vectors, dtype="float32")
        self._faiss_index.add(arr)
        self._faiss_ids.extend(ids)

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        cleaned_vectors: List[List[float]] = []
        for idx, vec in zip(ids, vectors):
            cleaned = self._ensure_dim(vec)
            self._id_to_vector[idx] = cleaned
            cleaned_vectors.append(cleaned)
        if self._use_faiss and faiss is not None:
            try:
                import numpy as np  # type: ignore
            except Exception:
                self._use_faiss = False
                self._faiss_index = None
                self._faiss_ids = []
                return
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatL2(self.dim)  # type: ignore[attr-defined]
            arr = np.asarray(cleaned_vectors, dtype="float32")
            if arr.size > 0:
                self._faiss_index.add(arr)
                self._faiss_ids.extend(ids)

    def delete(self, ids: Sequence[str]) -> None:
        for idx in ids:
            self._id_to_vector.pop(idx, None)
        self._rebuild_faiss()

    def search(self, vector: Sequence[float], k: int) -> List[Tuple[str, float]]:
        cleaned_vector = self._ensure_dim(vector)
        if self._use_faiss and self._faiss_index is not None and self._faiss_ids:
            try:
                import numpy as np  # type: ignore
            except Exception:
                self._use_faiss = False
            else:
                query = np.asarray([cleaned_vector], dtype="float32")
                k_eff = min(k, len(self._faiss_ids)) if k > 0 else len(self._faiss_ids)
                distances, indices = self._faiss_index.search(query, k_eff)
                results: List[Tuple[str, float]] = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx == -1:
                        continue
                    if idx < len(self._faiss_ids):
                        results.append((self._faiss_ids[idx], float(dist)))
                if results:
                    return results
        results: List[Tuple[str, float]] = []
        target = cleaned_vector
        for idx, vec in self._id_to_vector.items():
            dist = sum((a - b) ** 2 for a, b in zip(vec, target))
            results.append((idx, dist))
        results.sort(key=lambda x: x[1])
        return results[:k]

    def count(self) -> int:
        return len(self._id_to_vector)
