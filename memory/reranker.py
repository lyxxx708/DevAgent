from __future__ import annotations

from typing import List

from pydantic import BaseModel

from infra.observer import UnifiedObserver
from schemas.memory import MemoryItem
from schemas.meta import RerankHints


class RankingResult(BaseModel):
    ranked_ids: List[str]


class MemoryReranker:
    """Baseline reranker combining recency and simple dimension boosts.

    - ``prefer_recent`` uses ``stats['created_at']`` (when present) to favor newer items.
    - ``boost_dimensions`` adds the provided weight when a key exists in ``item.dimensions``.
    - ``diversity_over`` is currently ignored; diversity handling is not implemented in this baseline.
    """

    def __init__(self, observer: UnifiedObserver | None = None, min_observer_items: int = 2) -> None:
        self.observer = observer
        self.min_observer_items = min_observer_items

    def _heuristic_rerank(self, items: List[MemoryItem], hints: RerankHints | None = None) -> List[MemoryItem]:
        if hints is None:
            return items

        scored: list[tuple[float, MemoryItem]] = []
        for item in items:
            score = 0.0
            if hints.prefer_recent:
                created_at = 0.0
                if isinstance(item.stats, dict):
                    created_at = float(item.stats.get("created_at", 0.0))
                score += created_at
            if hints.boost_dimensions:
                for key, weight in hints.boost_dimensions.items():
                    if key in item.dimensions:
                        score += weight
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _score, item in scored]

    def rerank(self, items: List[MemoryItem], hints: RerankHints | None = None) -> List[MemoryItem]:
        if self.observer is None or len(items) < self.min_observer_items:
            return self._heuristic_rerank(items, hints=hints)

        try:
            context_items = [
                {"id": item.id, "kind": item.kind, "snippet": item.snippet}
                for item in items
            ]
            result = self.observer.perceive({"items": context_items}, response_model=RankingResult)
            ranked_ids = result.ranked_ids or []
            rank_map = {item_id: idx for idx, item_id in enumerate(ranked_ids)}
            ranked = [item for item in items if item.id in rank_map]
            ranked.sort(key=lambda item: rank_map.get(item.id, len(rank_map)))
            remaining = [item for item in items if item.id not in rank_map]
            return ranked + remaining
        except Exception:
            return self._heuristic_rerank(items, hints=hints)
