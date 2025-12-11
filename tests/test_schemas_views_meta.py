from schemas.memory import MemoryItem, MemoryStats
from schemas.meta import (
    FocusSpec,
    GoalViewSummary,
    MetaInputView,
    MetaPlan,
    RerankHints,
    SelectorProfile,
    StateSummary,
    TraceHint,
)
from schemas.views import (
    AgentHints,
    DecisionInputView,
    DevAgentMode,
    FocusView,
    GoalView,
    MemoryView,
    StateView,
)


def test_decision_input_view_validation():
    state_view = StateView(git_head="abc", failing_tests=["tests/test_a.py"], repo_stats={"files": 1})
    focus_view = FocusView(files=["a.py"], modules=["mod"], tests=["tests/test_a.py"])
    memory_item = MemoryItem(
        id="m1",
        kind="error_pattern",
        pointer={"file": "a.py"},
        snippet="error...",
        dimensions={"layer": "L2"},
        stats={"score": 0.4},
    )
    memory_view = MemoryView(items=[memory_item], stats=MemoryStats(counts_by_kind={"error_pattern": 1}, recent_activity_score=0.8))
    goal_view = GoalView(task_type="fix_failures", natural_language_goal="让 pytest 全部通过")
    hints = AgentHints(last_step_outcome="ok", consecutive_no_progress=0)
    view = DecisionInputView(
        state_view=state_view,
        focus_view=focus_view,
        memory_view=memory_view,
        goal_view=goal_view,
        hints=hints,
        mode=DevAgentMode.OPTIMIZED_STRUCTURED,
        token_budget_hint=4096,
    )

    assert view.mode == DevAgentMode.OPTIMIZED_STRUCTURED
    assert view.goal_view.task_type == "fix_failures"


def test_meta_plan_and_input():
    focus_spec = FocusSpec(task_type="fix_failures", modules=["mod1"], max_focus_files=10)
    selector_profile = SelectorProfile(weights={"recent": 1.0}, per_kind_limit={"error_pattern": 5})
    rerank_hints = RerankHints(boost_dimensions={"layer": 1.0}, diversity_over=["module"], prefer_recent=True)
    meta_plan = MetaPlan(
        focus_spec=focus_spec,
        selector_profile=selector_profile,
        rerank_hints=rerank_hints,
    )

    meta_input = MetaInputView(
        goal_view=GoalViewSummary(task_type="fix_failures", natural_language_goal="让 pytest 全部通过"),
        state_summary=StateSummary(repo_size=1234, failing_tests_count=2, key_modules=["core"]),
        memory_stats=MemoryStats(counts_by_kind={"error_pattern": 1}, recent_activity_score=0.9),
        trace_hint=TraceHint(recent_steps=2, last_status="ok"),
        mode="bootstrap_llm_heavy",
    )

    assert meta_plan.focus_spec.task_type == "fix_failures"
    assert meta_input.mode == "bootstrap_llm_heavy"
