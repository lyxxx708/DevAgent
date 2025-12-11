from __future__ import annotations

from typing import Iterable

from schemas.core import Event
from store.event_store import EventStore
from store.trace_ledger import TraceEntry, TraceLedger


class UnifiedObserver:
    def __init__(self, event_store: EventStore | None = None, trace_ledger: TraceLedger | None = None) -> None:
        self.event_store = event_store
        self.trace_ledger = trace_ledger

    def record_events(self, events: Iterable[Event]) -> None:
        if self.event_store is not None:
            self.event_store.append(events)

    def record_trace(self, entry: TraceEntry) -> None:
        if self.trace_ledger is not None:
            self.trace_ledger.append(entry)


class NullObserver:
    def record_events(self, events: Iterable[Event]) -> None:  # noqa: ARG002
        return None

    def record_trace(self, entry: TraceEntry) -> None:  # noqa: ARG002
        return None
