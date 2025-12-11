import tempfile

from schemas.core import Event
from memory.store import MemoryStore
from memory.ingest import MemoryIngestPipeline


def test_memory_store_upsert_and_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(db_path=f"{tmpdir}/memory.db")
        item_event = Event(
            event_id="e1",
            job_id="j1",
            step_id=1,
            type="RUN",
            payload={"exit_code": 1, "stderr": "traceback in src/module.py"},
            started_at=0.0,
            ended_at=0.1,
        )
        ingest = MemoryIngestPipeline(store)
        ingest.ingest([item_event])
        stats = store.stats()
        assert stats.counts_by_kind.get("error_pattern", 0) == 1


def test_memory_store_query_by_dimensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(db_path=f"{tmpdir}/memory.db")
        event = Event(
            event_id="e2",
            job_id="j1",
            step_id=2,
            type="EDIT",
            payload={"file_path": "src/app.py"},
            started_at=0.0,
            ended_at=0.1,
        )
        ingest = MemoryIngestPipeline(store)
        ingest.ingest([event])
        results = store.query_by_dimensions({"file_path": "src/app.py"})
        assert len(results) == 1
        assert results[0].dimensions["file_path"] == "src/app.py"


def test_memory_ingest_defaults_and_run_success_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(db_path=f"{tmpdir}/memory.db")
        ingest = MemoryIngestPipeline(store)

        success_event = Event(
            event_id="run-success",
            job_id="job",
            step_id=1,
            type="RUN",
            payload={"stderr": "no issues"},
            started_at=0.0,
            ended_at=0.1,
        )
        failure_event = Event(
            event_id="run-fail",
            job_id="job",
            step_id=2,
            type="RUN",
            payload={"exit_code": 2, "stderr": "error in src/main.py"},
            started_at=0.2,
            ended_at=0.3,
        )

        ingest.ingest([success_event, failure_event])

        success_item = store.get_item("run-success")
        assert success_item is not None
        assert success_item.kind == "run_config"
        assert success_item.dimensions["exit_code"] == 0

        failure_item = store.get_item("run-fail")
        assert failure_item is not None
        assert failure_item.kind == "error_pattern"
        assert "exit_code=2" in failure_item.snippet
        assert failure_item.dimensions["exit_code"] == 2
