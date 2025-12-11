from typing import Literal

from pydantic import BaseModel, Field

from schemas.memory import MemoryStats


class GoalViewSummary(BaseModel):
    task_type: str
    natural_language_goal: str


class StateSummary(BaseModel):
    repo_size: int
    failing_tests_count: int
    key_modules: list[str]


class TraceHint(BaseModel):
    recent_steps: int
    last_status: Literal["ok", "stuck", "flaky"]


class MetaInputView(BaseModel):
    goal_view: GoalViewSummary
    state_summary: StateSummary
    memory_stats: MemoryStats
    trace_hint: TraceHint | None = None
    mode: Literal["bootstrap_llm_heavy", "optimized_structured"]


class FocusSpec(BaseModel):
    task_type: str
    modules: list[str] = Field(default_factory=list)
    only_failing_tests: bool = True
    max_focus_files: int = 20


class SelectorProfile(BaseModel):
    weights: dict[str, float]
    per_kind_limit: dict[str, int]
    recency_window: int | None = None


class RerankHints(BaseModel):
    boost_dimensions: dict[str, float] | None = None
    diversity_over: list[str] | None = None
    prefer_recent: bool = True


class MetaPlan(BaseModel):
    focus_spec: FocusSpec
    selector_profile: SelectorProfile
    rerank_hints: RerankHints | None = None


__all__ = [
    "GoalViewSummary",
    "StateSummary",
    "TraceHint",
    "MetaInputView",
    "FocusSpec",
    "SelectorProfile",
    "RerankHints",
    "MetaPlan",
]
