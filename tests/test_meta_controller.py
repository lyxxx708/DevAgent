from __future__ import annotations

import tempfile
from pathlib import Path

from agent.devagent import DevAgent
from infra.observer import UnifiedObserver
from meta.controller import MetaController
from meta.llm_meta_planner import LLMMetaPlanner
from schemas.core import Event, Instruction, Program, State
from schemas.views import AgentHints, DevAgentMode, GoalView
from store.event_store import EventStore
from store.trace_ledger import TraceLedger
from memory.store import MemoryStore
from schemas.meta import MetaInputView, MetaPlan


def test_meta_controller_run_step_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        event_store = EventStore(db_path=str(base / "events.db"))
        trace_ledger = TraceLedger(db_path=str(base / "trace.db"))
        memory_store = MemoryStore(db_path=str(base / "memory.db"))
        observer = UnifiedObserver(event_store=event_store, trace_ledger=trace_ledger)

        devagent = DevAgent(
            mode=DevAgentMode.OPTIMIZED_STRUCTURED,
            observer=observer,
            memory_store=memory_store,
            vector_store=None,
        )
        planner = LLMMetaPlanner()
        controller = MetaController(
            devagent=devagent,
            planner=planner,
            memory_store=memory_store,
            trace_ledger=trace_ledger,
            event_store=event_store,
        )

        program = Program(
            instructions=[
                Instruction(kind="RUN", payload={"cmd": 'python -c "import sys; sys.exit(1)"'}),
            ]
        )
        state = State(git_head="", repo_root=str(base))
        goal_view = GoalView(task_type="fix_failures", natural_language_goal="fix failing tests")

        new_state, events, decision_input = controller.run_step(
            job_id="job-1",
            state=state,
            program=program,
            goal_view=goal_view,
            hints=AgentHints(),
        )

        assert new_state.git_head is not None
        assert events
        assert decision_input.mode == DevAgentMode.OPTIMIZED_STRUCTURED
        assert decision_input.goal_view.natural_language_goal == "fix failing tests"

        recent = trace_ledger.recent_for_job("job-1")
        assert recent
        assert recent[0].job_id == "job-1"
        assert recent[0].step_id >= 1


def test_meta_controller_state_summary_uses_event_store() -> None:
    class CapturingPlanner(LLMMetaPlanner):
        def __init__(self) -> None:
            super().__init__()
            self.captured_meta_input: MetaInputView | None = None

        def propose_plan(self, meta_input: MetaInputView) -> MetaPlan:  # type: ignore[override]
            self.captured_meta_input = meta_input
            return super().propose_plan(meta_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        event_store = EventStore(db_path=str(base / "events.db"))
        trace_ledger = TraceLedger(db_path=str(base / "trace.db"))
        memory_store = MemoryStore(db_path=str(base / "memory.db"))
        observer = UnifiedObserver(event_store=event_store, trace_ledger=trace_ledger)

        failing_event = Event(
            event_id="evt-1",
            job_id="job-1",
            step_id=1,
            type="RUN",
            payload={"exit_code": 1, "stderr": "E   AssertionError: fail"},
            started_at=0.0,
            ended_at=0.1,
        )
        event_store.append([failing_event])

        devagent = DevAgent(
            mode=DevAgentMode.OPTIMIZED_STRUCTURED,
            observer=observer,
            memory_store=memory_store,
            vector_store=None,
        )

        planner = CapturingPlanner()
        controller = MetaController(
            devagent=devagent,
            planner=planner,
            memory_store=memory_store,
            trace_ledger=trace_ledger,
            event_store=event_store,
        )

        program = Program(instructions=[])
        state = State(git_head="", repo_root=str(base))
        goal_view = GoalView(task_type="fix_failures", natural_language_goal="fix tests")

        controller.run_step(
            job_id="job-1",
            state=state,
            program=program,
            goal_view=goal_view,
            hints=AgentHints(),
        )

        meta_input = planner.captured_meta_input
        assert meta_input is not None
        assert meta_input.state_summary.failing_tests_count >= 1
