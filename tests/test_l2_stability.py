from __future__ import annotations

from pathlib import Path
import tempfile

from agent.devagent import DevAgent
from infra.observer import UnifiedObserver
from memory.store import MemoryStore
from meta.controller import MetaController
from meta.llm_meta_planner import LLMMetaPlanner
from schemas.core import Instruction, Program
from schemas.views import DevAgentMode, GoalView
from store.event_store import EventStore
from store.trace_ledger import TraceLedger
from task.runner import TaskRunner


def test_repeated_steps_stay_stable():
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
        )
        runner = TaskRunner(controller=controller)

        goal_view = GoalView(task_type="fix_failures", natural_language_goal="goal")
        program = Program(instructions=[Instruction(kind="RUN", payload={"cmd": 'python -c "import sys; sys.exit(1)"'})])

        job_id = runner.create_job(repo_root=str(base), goal_view=goal_view)

        event_counts: list[int] = []
        trace_counts: list[int] = []
        memory_totals: list[int] = []

        for idx in range(3):
            _, events, _ = runner.run_step(job_id=job_id, program=program)
            event_counts.append(len(event_store.recent_for_job(job_id)))
            trace_counts.append(len(trace_ledger.recent_for_job(job_id)))
            stats = memory_store.stats()
            memory_totals.append(sum(stats.counts_by_kind.values()))
            if events:
                assert max(evt.step_id for evt in events) >= idx + 1

        assert event_counts == sorted(event_counts)
        assert trace_counts == sorted(trace_counts)
        assert memory_totals == sorted(memory_totals)
