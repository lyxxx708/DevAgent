from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

from schemas.memory import MemoryItem, MemoryStats


class ViewSpec(BaseModel):
    kind: Literal["state", "focus", "memory", "trace", "ui"]
    params: dict[str, Any]


class StateView(BaseModel):
    git_head: str
    failing_tests: list[str]
    repo_stats: dict[str, Any]


class FocusView(BaseModel):
    files: list[str]
    modules: list[str]
    tests: list[str]


class MemoryView(BaseModel):
    items: list[MemoryItem]
    stats: MemoryStats | None = None


class GoalView(BaseModel):
    task_type: Literal["init_project", "fix_failures", "run_experiments", "upgrade_repo"]
    natural_language_goal: str


class AgentHints(BaseModel):
    last_step_outcome: str | None = None
    consecutive_no_progress: int = 0


class DevAgentMode(str, Enum):
    OPTIMIZED_STRUCTURED = "optimized_structured"
    BOOTSTRAP_LLM_HEAVY = "bootstrap_llm_heavy"


class DecisionInputView(BaseModel):
    state_view: StateView
    focus_view: FocusView
    memory_view: MemoryView
    goal_view: GoalView
    hints: AgentHints
    mode: DevAgentMode
    token_budget_hint: int | None = None


__all__ = [
    "ViewSpec",
    "StateView",
    "FocusView",
    "MemoryView",
    "GoalView",
    "AgentHints",
    "DevAgentMode",
    "DecisionInputView",
]
