from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from agent.devagent import DevAgent
from infra.observer import UnifiedObserver
from infra.vector_store import VectorStore
from memory.store import MemoryStore
from meta.controller import MetaController
from meta.llm_meta_planner import LLMMetaPlanner
from schemas.core import Instruction, Program
from schemas.views import DevAgentMode, GoalView
from store.event_store import EventStore
from store.trace_ledger import TraceLedger
from task.runner import TaskRunner


def build_stack(base_dir: Path) -> TaskRunner:
    """Construct the offline DevAgent stack using local SQLite files in base_dir."""

    event_store = EventStore(db_path=str(base_dir / "events.db"))
    trace_ledger = TraceLedger(db_path=str(base_dir / "trace.db"))
    memory_store = MemoryStore(db_path=str(base_dir / "memory.db"))
    observer = UnifiedObserver(event_store=event_store, trace_ledger=trace_ledger)
    vector_store = VectorStore(dim=3, use_faiss=False)
    devagent = DevAgent(
        mode=DevAgentMode.OPTIMIZED_STRUCTURED,
        observer=observer,
        memory_store=memory_store,
        vector_store=vector_store,
    )
    planner = LLMMetaPlanner()
    meta_controller = MetaController(
        devagent=devagent,
        planner=planner,
        memory_store=memory_store,
        trace_ledger=trace_ledger,
    )
    return TaskRunner(controller=meta_controller)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single DevAgent v7.3 step locally without HTTP.")
    parser.add_argument("repo_root", help="Path to the repository root to operate on")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    goal_view = GoalView(task_type="fix_failures", natural_language_goal="Make tests pass for this repo")
    program = Program(
        instructions=[
            Instruction(kind="RUN", payload={"cmd": 'python -c "import sys; sys.exit(1)"'}),
        ]
    )

    with tempfile.TemporaryDirectory(prefix="devagent_local_") as tmp:
        base_dir = Path(tmp)
        runner = build_stack(base_dir)

        job_id = runner.create_job(repo_root=str(repo_root), goal_view=goal_view)
        state, events, decision = runner.run_step(job_id=job_id, program=program)

        focus_files = decision.focus_view.files
        memory_items = decision.memory_view.items
        memory_stats = decision.memory_view.stats

        print("DevAgent v7.3 local run summary")
        print("Job ID:", job_id)
        print("Repo root:", repo_root)
        print("Git HEAD:", state.git_head)
        print("Event count:", len(events))
        print("Mode:", decision.mode.value)
        print("Goal:", decision.goal_view.natural_language_goal)
        print("Focus files (sample):", focus_files[:5])
        print("Memory items:", len(memory_items))
        if memory_stats:
            print("Memory counts by kind:", memory_stats.counts_by_kind)


if __name__ == "__main__":
    main()
