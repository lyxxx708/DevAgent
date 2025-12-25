from __future__ import annotations

from schemas.memory import MemoryStats
from infra.observer import UnifiedObserver
from schemas.meta import (
    FocusSpec,
    GoalViewSummary,
    MetaInputView,
    RerankHints,
    SelectorProfile,
    StateSummary,
    TraceHint,
)
from meta.llm_meta_planner import LLMMetaPlanner


def test_meta_planner_generates_plan_for_fix_failures() -> None:
    planner = LLMMetaPlanner()
    meta_input = MetaInputView(
        goal_view=GoalViewSummary(task_type="fix_failures", natural_language_goal="Fix failing tests"),
        state_summary=StateSummary(repo_size=10, failing_tests_count=2, key_modules=["core", "memory"]),
        memory_stats=MemoryStats(counts_by_kind={"error_pattern": 3}, recent_activity_score=8.0),
        trace_hint=TraceHint(recent_steps=5, last_status="ok"),
        mode="optimized_structured",
    )

    plan = planner.propose_plan(meta_input)

    assert plan.focus_spec.task_type == "fix_failures"
    assert plan.focus_spec.only_failing_tests is True
    assert plan.focus_spec.max_focus_files >= 1
    assert "error_pattern" in plan.selector_profile.per_kind_limit
    assert "run_config" in plan.selector_profile.per_kind_limit
    assert plan.rerank_hints is not None
    assert plan.rerank_hints.prefer_recent is True


def test_meta_planner_defaults_for_init_project() -> None:
    planner = LLMMetaPlanner()
    meta_input = MetaInputView(
        goal_view=GoalViewSummary(task_type="init_project", natural_language_goal="Start new project"),
        state_summary=StateSummary(repo_size=0, failing_tests_count=0, key_modules=[]),
        memory_stats=MemoryStats(counts_by_kind={}, recent_activity_score=0.5),
        trace_hint=None,
        mode="bootstrap_llm_heavy",
    )

    plan = planner.propose_plan(meta_input)

    assert plan.focus_spec.task_type == "init_project"
    assert plan.focus_spec.only_failing_tests is False
    assert plan.focus_spec.max_focus_files == 10
    assert plan.selector_profile.per_kind_limit.get("error_pattern") == 10
    assert plan.selector_profile.per_kind_limit.get("run_config") == 10
    assert plan.rerank_hints is not None
    assert plan.rerank_hints.prefer_recent is True


def test_meta_planner_uses_llm_response_when_available() -> None:
    captured_payload: dict[str, object] = {}

    def perceiver(payload: dict[str, object]) -> dict[str, object]:
        captured_payload.update(payload)
        return {
            "focus_spec": FocusSpec(
                task_type="custom_plan",
                modules=["meta"],
                only_failing_tests=False,
                max_focus_files=5,
            ).model_dump(),
            "selector_profile": SelectorProfile(
                weights={"error_pattern": 0.5},
                per_kind_limit={"error_pattern": 3},
                recency_window=None,
            ).model_dump(),
            "rerank_hints": RerankHints(
                boost_dimensions={"layer": 0.5},
                diversity_over=["module"],
                prefer_recent=False,
            ).model_dump(),
        }

    observer = UnifiedObserver(perceiver=perceiver)
    planner = LLMMetaPlanner(observer=observer)
    meta_input = MetaInputView(
        goal_view=GoalViewSummary(task_type="fix_failures", natural_language_goal="Fix failing tests"),
        state_summary=StateSummary(repo_size=10, failing_tests_count=2, key_modules=["core", "memory"]),
        memory_stats=MemoryStats(counts_by_kind={"error_pattern": 3}, recent_activity_score=8.0),
        trace_hint=TraceHint(recent_steps=5, last_status="ok"),
        mode="optimized_structured",
    )

    plan = planner.propose_plan(meta_input)

    assert captured_payload["kind"] == "meta_plan_request"
    assert "meta_input" in captured_payload
    assert plan.focus_spec.task_type == "custom_plan"
    assert plan.selector_profile.per_kind_limit["error_pattern"] == 3
    assert plan.rerank_hints is not None
    assert plan.rerank_hints.prefer_recent is False
