from __future__ import annotations

from typing import Any, Callable, Iterable

from schemas.core import Event
from store.event_store import EventStore
from store.trace_ledger import TraceEntry, TraceLedger


Perceiver = Callable[[dict[str, Any]], dict[str, Any] | None]


class UnifiedObserver:
    def __init__(
        self,
        event_store: EventStore | None = None,
        trace_ledger: TraceLedger | None = None,
        perceiver: Perceiver | None = None,
    ) -> None:
        self.event_store = event_store
        self.trace_ledger = trace_ledger
        self.perceiver = perceiver

    def record_events(self, events: Iterable[Event]) -> None:
        if self.event_store is not None:
            self.event_store.append(events)

    def record_trace(self, entry: TraceEntry) -> None:
        if self.trace_ledger is not None:
            self.trace_ledger.append(entry)

    def perceive(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self.perceiver is None:
            return None
        return self.perceiver(payload)


class NullObserver:
    def record_events(self, events: Iterable[Event]) -> None:  # noqa: ARG002
        return None

    def record_trace(self, entry: TraceEntry) -> None:  # noqa: ARG002
        return None

    def perceive(self, payload: dict[str, Any]) -> None:  # noqa: ARG002
        return None
