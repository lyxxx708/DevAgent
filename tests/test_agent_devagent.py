from __future__ import annotations

import tempfile

from agent.devagent import DevAgent
from infra.observer import UnifiedObserver
from schemas.core import Instruction, Program, State
from schemas.views import DevAgentMode, GoalView
from memory.store import MemoryStore
from infra.vector_store import VectorStore
from store.event_store import EventStore
from store.trace_ledger import TraceLedger


def test_devagent_run_step_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        event_db = f"{tmpdir}/events.db"
        trace_db = f"{tmpdir}/trace.db"
        memory_db = f"{tmpdir}/memory.db"

        event_store = EventStore(db_path=event_db)
        trace_ledger = TraceLedger(db_path=trace_db)
        memory_store = MemoryStore(db_path=memory_db)
        observer = UnifiedObserver(event_store=event_store, trace_ledger=trace_ledger)
        vector_store = VectorStore(dim=3, use_faiss=False)

        agent = DevAgent(
            mode=DevAgentMode.OPTIMIZED_STRUCTURED,
            observer=observer,
            memory_store=memory_store,
            vector_store=vector_store,
        )

        program = Program(
            instructions=[
                Instruction(kind="RUN", payload={"cmd": 'python -c "import sys; sys.exit(1)"'}),
            ]
        )
        state = State(git_head="", repo_root=tmpdir)
        goal_view = GoalView(task_type="fix_failures", natural_language_goal="Make tests pass")

        new_state, events, decision_input = agent.run_step(
            job_id="job-1",
            state=state,
            program=program,
            goal_view=goal_view,
        )

        assert isinstance(new_state, State)
        assert events
        assert decision_input.mode == DevAgentMode.OPTIMIZED_STRUCTURED
        assert decision_input.goal_view.natural_language_goal == "Make tests pass"

        persisted_events = event_store.recent_for_job("job-1")
        assert len(persisted_events) == len(events)

        stats = memory_store.stats()
        assert sum(stats.counts_by_kind.values()) >= 1
