from __future__ import annotations

from pathlib import Path
import tempfile

from agent.devagent import DevAgent
from infra.observer import UnifiedObserver
from infra.vector_store import VectorStore
from memory.store import MemoryStore
from meta.controller import MetaController
from meta.llm_meta_planner import LLMMetaPlanner
from schemas.core import Instruction, Program, State
from schemas.views import AgentHints, DevAgentMode, GoalView
from store.event_store import EventStore
from store.trace_ledger import TraceLedger
from task.runner import TaskRunner


def test_task_runner_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        event_store = EventStore(db_path=str(base / "events.db"))
        trace_ledger = TraceLedger(db_path=str(base / "trace.db"))
        memory_store = MemoryStore(db_path=str(base / "memory.db"))
        observer = UnifiedObserver(event_store=event_store, trace_ledger=trace_ledger)
        vector_store = VectorStore(dim=3, use_faiss=False)

        devagent = DevAgent(
            mode=DevAgentMode.OPTIMIZED_STRUCTURED,
            observer=observer,
            memory_store=memory_store,
            vector_store=vector_store,
        )
        planner = LLMMetaPlanner()
        controller = MetaController(
            devagent=devagent,
            planner=planner,
            memory_store=memory_store,
            trace_ledger=trace_ledger,
            event_store=event_store,
        )

        runner = TaskRunner(controller=controller)

        goal_view = GoalView(task_type="fix_failures", natural_language_goal="Make tests pass")
        program = Program(
            instructions=[
                Instruction(kind="RUN", payload={"cmd": 'python -c "import sys; sys.exit(1)"'}),
            ]
        )

        job_id = runner.create_job(repo_root=str(base), goal_view=goal_view)
        new_state, events, decision_input = runner.run_step(job_id=job_id, program=program)

        assert job_id
        assert isinstance(new_state, State)
        assert events
        assert decision_input.mode == DevAgentMode.OPTIMIZED_STRUCTURED
        assert decision_input.goal_view.natural_language_goal == "Make tests pass"
        assert isinstance(decision_input.focus_view.files, list)
        assert len(decision_input.memory_view.items) >= 1

        recent_events = event_store.recent_for_job(job_id)
        assert len(recent_events) >= 1

        trace_entries = trace_ledger.recent_for_job(job_id)
        assert len(trace_entries) >= 1
        assert trace_entries[0].job_id == job_id

        stats = memory_store.stats()
        assert sum(stats.counts_by_kind.values()) >= 1

        # run a second step to ensure step ids advance and state persists
        second_program = Program(
            instructions=[
                Instruction(kind="RUN", payload={"cmd": 'python -c "import sys; sys.exit(1)"'}),
            ]
        )
        _, more_events, _ = runner.run_step(job_id=job_id, program=second_program, hints=AgentHints())
        assert more_events
        assert max(evt.step_id for evt in more_events) >= 2
