from __future__ import annotations

import uuid
from typing import Any

from agent.devagent import DevAgent
from memory.store import MemoryStore
from meta.llm_meta_planner import LLMMetaPlanner
from schemas.core import Event, Program, State
from schemas.meta import GoalViewSummary, MetaInputView, StateSummary
from schemas.views import AgentHints, DecisionInputView, DevAgentMode, GoalView
from store.trace_ledger import TraceEntry, TraceLedger


class MetaController:
    """Meta-level orchestrator that wraps DevAgent with planning and tracing."""

    def __init__(
        self,
        devagent: DevAgent,
        planner: LLMMetaPlanner,
        memory_store: MemoryStore,
        trace_ledger: TraceLedger,
    ) -> None:
        self.devagent = devagent
        self.planner = planner
        self.memory_store = memory_store
        self.trace_ledger = trace_ledger

    def _mode_literal(self, mode: DevAgentMode) -> str:
        if mode == DevAgentMode.BOOTSTRAP_LLM_HEAVY:
            return "bootstrap_llm_heavy"
        return "optimized_structured"

    def run_step(
        self,
        job_id: str,
        state: State,
        program: Program,
        goal_view: GoalView,
        *,
        hints: AgentHints | None = None,
        start_step_id: int = 1,
    ) -> tuple[State, list[Event], DecisionInputView]:
        """Plan and execute a single step: build MetaInputView, call planner, delegate to DevAgent, and trace the outcome."""
        goal_summary = GoalViewSummary(
            task_type=goal_view.task_type,
            natural_language_goal=goal_view.natural_language_goal,
        )
        state_summary = StateSummary(
            repo_size=0,
            failing_tests_count=0,
            key_modules=[],
        )
        memory_stats = self.memory_store.stats()

        meta_input = MetaInputView(
            goal_view=goal_summary,
            state_summary=state_summary,
            memory_stats=memory_stats,
            trace_hint=None,
            mode=self._mode_literal(self.devagent.mode),
        )

        plan = self.planner.propose_plan(meta_input)

        new_state, events, decision_input = self.devagent.run_step(
            job_id=job_id,
            state=state,
            program=program,
            goal_view=goal_view,
            hints=hints,
            focus_spec=plan.focus_spec,
            selector_profile=plan.selector_profile,
            start_step_id=start_step_id,
            rerank_hints=plan.rerank_hints,
        )

        decision_id = str(uuid.uuid4())
        step_id = max((event.step_id for event in events), default=start_step_id)
        decision_input_summary: dict[str, Any] = {
            "goal_task_type": decision_input.goal_view.task_type,
            "files": decision_input.focus_view.files,
            "mode": decision_input.mode,
        }
        program_summary = {"instruction_count": len(program.instructions)}
        outcome_summary = {
            "event_count": len(events),
            "git_head": new_state.git_head,
            "rerank_hints": plan.rerank_hints.model_dump() if plan.rerank_hints else None,
        }

        entry = TraceEntry(
            decision_id=decision_id,
            job_id=job_id,
            step_id=step_id,
            decision_input_summary=decision_input_summary,
            program_summary=program_summary,
            outcome_summary=outcome_summary,
        )
        self.trace_ledger.append(entry)

        return new_state, events, decision_input
