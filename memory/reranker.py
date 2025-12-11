from __future__ import annotations

from typing import List

from schemas.memory import MemoryItem
from schemas.meta import RerankHints


class MemoryReranker:
    """Baseline reranker combining recency and simple dimension boosts.

    - ``prefer_recent`` uses ``stats['created_at']`` (when present) to favor newer items.
    - ``boost_dimensions`` adds the provided weight when a key exists in ``item.dimensions``.
    - ``diversity_over`` is currently ignored; diversity handling is not implemented in this baseline.
    """

    def rerank(self, items: List[MemoryItem], hints: RerankHints | None = None) -> List[MemoryItem]:
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
