from __future__ import annotations

import os
import tempfile

from agent.devagent import DevAgent
from infra.observer import NullObserver
from memory.store import MemoryStore
from schemas.core import Program, State
from schemas.memory import MemoryItem
from schemas.meta import RerankHints
from schemas.views import DevAgentMode, GoalView


def test_rerank_hints_boost_dimensions_affect_focus_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(db_path=os.path.join(tmpdir, "memory.db"))
        layered = MemoryItem(
            id="layered",
            kind="run_config",
            pointer={},
            snippet="",
            dimensions={"file_path": "src/layered.py", "layer": "L2"},
            stats={"created_at": 1.0},
        )
        plain = MemoryItem(
            id="plain",
            kind="run_config",
            pointer={},
            snippet="",
            dimensions={"file_path": "src/plain.py"},
            stats={"created_at": 1.0},
        )
        store.upsert_item(layered)
        store.upsert_item(plain)

        devagent = DevAgent(
            mode=DevAgentMode.OPTIMIZED_STRUCTURED,
            observer=NullObserver(),
            memory_store=store,
            vector_store=None,
        )

        goal_view = GoalView(task_type="fix_failures", natural_language_goal="goal")
        program = Program(instructions=[])

        _, _, decision = devagent.run_step(
            job_id="job",
            state=State(git_head="", repo_root=tmpdir),
            program=program,
            goal_view=goal_view,
            rerank_hints=RerankHints(boost_dimensions={"layer": 10.0}, diversity_over=None, prefer_recent=False),
        )

        assert decision.focus_view.files
        assert decision.focus_view.files[0] == "src/layered.py"
