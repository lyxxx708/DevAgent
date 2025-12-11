from __future__ import annotations

import os
import tempfile

from schemas.core import Event
from store.event_store import EventStore
from store.trace_ledger import TraceEntry, TraceLedger
from infra.observer import NullObserver, UnifiedObserver


def _sample_event(job_id: str, step_id: int) -> Event:
    return Event(
        event_id="evt-1",
        job_id=job_id,
        step_id=step_id,
        type="RUN",
        payload={"cmd": "echo hi", "stdout": "hi", "stderr": "", "exit_code": 0},
        started_at=0.0,
        ended_at=0.1,
    )


def test_unified_observer_records_events_and_trace() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = os.path.join(tmpdir, "events.db")
        trace_path = os.path.join(tmpdir, "trace.db")
        event_store = EventStore(db_path=events_path)
        trace_ledger = TraceLedger(db_path=trace_path)
        observer = UnifiedObserver(event_store=event_store, trace_ledger=trace_ledger)

        events = [_sample_event("job-1", 1)]
        trace_entry = TraceEntry(
            decision_id="dec-1",
            job_id="job-1",
            step_id=1,
            decision_input_summary={},
            program_summary={},
            outcome_summary={},
        )

        observer.record_events(events)
        observer.record_trace(trace_entry)

        recent_events = event_store.recent_for_job("job-1")
        recent_trace = trace_ledger.recent_for_job("job-1")

        assert len(recent_events) == 1
        assert recent_events[0].event_id == "evt-1"
        assert len(recent_trace) == 1
        assert recent_trace[0].decision_id == "dec-1"


def test_null_observer_is_noop() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = os.path.join(tmpdir, "events.db")
        trace_path = os.path.join(tmpdir, "trace.db")
        event_store = EventStore(db_path=events_path)
        trace_ledger = TraceLedger(db_path=trace_path)

        observer = NullObserver()
        observer.record_events([_sample_event("job-2", 1)])
        observer.record_trace(
            TraceEntry(
                decision_id="dec-2",
                job_id="job-2",
                step_id=1,
                decision_input_summary={},
                program_summary={},
                outcome_summary={},
            )
        )

        assert event_store.recent_for_job("job-2") == []
        assert trace_ledger.recent_for_job("job-2") == []
