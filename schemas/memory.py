from typing import Any, Literal

from pydantic import BaseModel


class MemoryItem(BaseModel):
    id: str
    kind: Literal[
        "error_pattern",
        "module_history",
        "upgrade_step",
        "human_feedback",
        "run_config",
    ]
    pointer: dict[str, Any]
    snippet: str
    dimensions: dict[str, Any]
    stats: dict[str, Any]


class MemoryStats(BaseModel):
    counts_by_kind: dict[str, int]
    recent_activity_score: float


__all__ = ["MemoryItem", "MemoryStats"]
