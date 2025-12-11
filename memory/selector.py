from __future__ import annotations

from typing import Any, List, Sequence

from memory.store import MemoryStore
from infra.vector_store import VectorStore
from schemas.memory import MemoryItem
from schemas.meta import SelectorProfile


class MemorySelector:
    """High-level retrieval over MemoryStore and optional VectorStore.

    Behavior:
    - ``profile.per_kind_limit``: max number of items per ``MemoryItem.kind``.
    - ``profile.recency_window``: minimum ``created_at`` timestamp; older items are skipped.
    - ``profile.weights``: per-kind importance scores; higher-weight kinds are preferred with recency as a
      tiebreaker, applied before per-kind limits/recency filters (weights bias ordering, not inclusion).
    - Vector search: if a query vector is provided and the vector store returns matches, candidates are
      reordered by vector rank while keeping non-indexed items after ranked ones; if it returns no matches,
      the original store candidates are preserved as a fallback before applying weights/recency.
    """

    def __init__(self, store: MemoryStore, vector_store: VectorStore | None = None) -> None:
        self.store = store
        self.vector_store = vector_store

    def select(
        self,
        profile: SelectorProfile,
        filters: dict[str, Any] | None = None,
        query_vector: Sequence[float] | None = None,
        limit: int | None = None,
    ) -> List[MemoryItem]:
        candidates = self.store.query_by_dimensions(filters or {}, limit=limit or 1000)
        if self.vector_store is not None and query_vector is not None and candidates:
            search_results = self.vector_store.search(query_vector, k=len(candidates))
            if search_results:
                rank_map = {item_id: rank for rank, (item_id, _dist) in enumerate(search_results)}
                with_vec = [item for item in candidates if item.id in rank_map]
                without_vec = [item for item in candidates if item.id not in rank_map]
                with_vec.sort(key=lambda item: rank_map.get(item.id, len(rank_map)))
                candidates = with_vec + without_vec

        index_map = {item.id: idx for idx, item in enumerate(candidates)}
        if profile.weights:
            weight_map = profile.weights

            def weight_key(item: MemoryItem) -> tuple[float, float, int]:
                kind_weight = float(weight_map.get(item.kind, 0.0))
                created_at = 0.0
                if isinstance(item.stats, dict):
                    created_at = float(item.stats.get("created_at", 0.0) or 0.0)
                return (-kind_weight, -created_at, index_map.get(item.id, len(candidates)))

            candidates = sorted(candidates, key=weight_key)

        per_kind_limit = profile.per_kind_limit or {}
        selected: List[MemoryItem] = []
        kind_counts: dict[str, int] = {}
        for item in candidates:
            if profile.recency_window is not None:
                created_at = float(item.stats.get("created_at", 0.0)) if isinstance(item.stats, dict) else 0.0
                if created_at < profile.recency_window:
                    continue
            count = kind_counts.get(item.kind, 0)
            limit_for_kind = per_kind_limit.get(item.kind)
            if limit_for_kind is not None and count >= limit_for_kind:
                continue
            kind_counts[item.kind] = count + 1
            selected.append(item)
            if limit is not None and len(selected) >= limit:
                break
        return selected
