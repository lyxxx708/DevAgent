from __future__ import annotations

from typing import Any

from core.interpret import interpret
from infra.observer import UnifiedObserver
from infra.vector_store import VectorStore
from memory.ingest import MemoryIngestPipeline
from memory.reranker import MemoryReranker
from memory.selector import MemorySelector
from memory.store import MemoryStore
from schemas.core import Event, Program, State
from schemas.meta import FocusSpec, RerankHints, SelectorProfile
from schemas.views import (
    AgentHints,
    DecisionInputView,
    DevAgentMode,
    FocusView,
    GoalView,
    MemoryView,
    StateView,
)
from views.focus import BaselineFocusInferer, FocusViewBuilder


class DevAgent:
    def __init__(
        self,
        mode: DevAgentMode,
        observer: UnifiedObserver,
        memory_store: MemoryStore,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.mode = mode
        self.observer = observer
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.ingest_pipeline = MemoryIngestPipeline(memory_store)
        self.selector = MemorySelector(store=memory_store, vector_store=vector_store)
        self.reranker = MemoryReranker()
        self.focus_builder = FocusViewBuilder(selector=self.selector, reranker=self.reranker)
        self.baseline_focus_inferer = BaselineFocusInferer()

    def run_step(
        self,
        job_id: str,
        state: State,
        program: Program,
        goal_view: GoalView,
        *,
        start_step_id: int = 1,
        hints: AgentHints | None = None,
        focus_spec: FocusSpec | None = None,
        selector_profile: SelectorProfile | None = None,
        extra_filters: dict[str, Any] | None = None,
        rerank_hints: RerankHints | None = None,
    ) -> tuple[State, list[Event], DecisionInputView]:
        """Run one DevAgent S/F/T cycle and return the new state, emitted events, and decision context."""
        new_state, events = interpret(state, program, job_id=job_id, step_id=start_step_id)
        self.observer.record_events(events)

        self.ingest_pipeline.ingest(events)

        state_view = StateView(
            git_head=new_state.git_head,
            failing_tests=[],
            repo_stats={},
        )

        baseline_focus = self.baseline_focus_inferer.infer(events)

        if focus_spec is None:
            focus_spec = FocusSpec(
                task_type=goal_view.task_type,
                modules=[],
                only_failing_tests=True,
                max_focus_files=20,
            )
        if selector_profile is None:
            selector_profile = SelectorProfile(weights={}, per_kind_limit={}, recency_window=None)

        memory_focus = self.focus_builder.build(
            spec=focus_spec,
            profile=selector_profile,
            filters=extra_filters,
            hints=rerank_hints,
        )

        combined_focus = FocusView(
            files=list(dict.fromkeys(baseline_focus.files + memory_focus.files)),
            modules=list(dict.fromkeys(baseline_focus.modules + memory_focus.modules)),
            tests=list(dict.fromkeys(baseline_focus.tests + memory_focus.tests)),
        )

        memory_candidates = self.selector.select(
            profile=selector_profile,
            filters=extra_filters,
            limit=50,
        )
        memory_items = self.reranker.rerank(memory_candidates, hints=rerank_hints)
        stats = self.memory_store.stats()
        memory_view = MemoryView(items=memory_items, stats=stats)

        decision_input = DecisionInputView(
            state_view=state_view,
            focus_view=combined_focus,
            memory_view=memory_view,
            goal_view=goal_view,
            hints=hints or AgentHints(),
            mode=self.mode,
            token_budget_hint=None,
        )

        return new_state, events, decision_input
