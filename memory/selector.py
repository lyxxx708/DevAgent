from __future__ import annotations

from typing import Any, List, Sequence

from memory.store import MemoryStore
from infra.vector_store import VectorStore
from schemas.memory import MemoryItem
from schemas.meta import SelectorProfile


class MemorySelector:
    """Select MemoryItems using store filtering, optional vector search, and per-kind caps.

    - ``profile.per_kind_limit`` places a maximum number of items allowed for each ``MemoryItem.kind``.
    - ``profile.recency_window`` is treated as a minimum ``created_at`` timestamp; items older than this
      value (``stats['created_at'] < recency_window``) are skipped as too old.
    - If vector search returns no matches, store-level candidates are preserved as a fallback.
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
                ranked_candidates = [item for item in candidates if item.id in rank_map]
                ranked_candidates.sort(key=lambda item: rank_map.get(item.id, len(rank_map)))
                candidates = ranked_candidates

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
