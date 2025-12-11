from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi.testclient import TestClient

from api.http import create_app
from config.settings import Settings
from schemas.core import Instruction, Program
from schemas.views import GoalView


def test_health_endpoint_exposes_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        custom_settings = Settings(
            llm_api_key="dummy",
            event_db_path=str(base / "events.db"),
            memory_db_path=str(base / "memory.db"),
            trace_db_path=str(base / "trace.db"),
        )
        app = create_app(settings=custom_settings)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "optimized_structured"


def test_create_job_and_run_step_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        custom_settings = Settings(
            llm_api_key="dummy",
            event_db_path=str(base / "events.db"),
            memory_db_path=str(base / "memory.db"),
            trace_db_path=str(base / "trace.db"),
        )
        app = create_app(settings=custom_settings)
        client = TestClient(app)

        goal_view = GoalView(task_type="fix_failures", natural_language_goal="Make tests pass")
        create_resp = client.post(
            "/jobs",
            json={"repo_root": str(base), "goal": goal_view.model_dump()},
        )
        assert create_resp.status_code == 200
        job_id = create_resp.json()["job_id"]
        assert job_id

        program = Program(
            instructions=[
                Instruction(kind="RUN", payload={"cmd": 'python -c "import sys; sys.exit(1)"'}),
            ]
        )

        step_resp = client.post(
            f"/jobs/{job_id}/steps",
            json={"program": program.model_dump(), "hints": None},
        )
        assert step_resp.status_code == 200
        data = step_resp.json()
        assert data["job_id"] == job_id
        assert data["decision"]["goal_view"]["natural_language_goal"] == "Make tests pass"
        assert data["decision"]["mode"] == "optimized_structured"
        assert isinstance(data["events"], list) and len(data["events"]) >= 1

        recent_events = app.state.event_store.recent_for_job(job_id)
        assert len(recent_events) >= 1
        stats = app.state.memory_store.stats()
        assert sum(stats.counts_by_kind.values()) >= 1


def test_run_step_unknown_job_returns_404():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        custom_settings = Settings(
            llm_api_key="dummy",
            event_db_path=str(base / "events.db"),
            memory_db_path=str(base / "memory.db"),
            trace_db_path=str(base / "trace.db"),
        )
        app = create_app(settings=custom_settings)
        client = TestClient(app)

        program = Program(
            instructions=[
                Instruction(kind="RUN", payload={"cmd": 'python -c "import sys; sys.exit(1)"'}),
            ]
        )

        resp = client.post(
            "/jobs/non-existent/steps",
            json={"program": program.model_dump(), "hints": None},
        )

        assert resp.status_code == 404
        assert resp.json().get("detail") == "job not found"
