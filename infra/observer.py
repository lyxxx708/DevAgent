from __future__ import annotations

from typing import Any, Iterable, TypeVar

from pydantic import BaseModel

import instructor
from openai import OpenAI
from pydantic import BaseModel

from config.settings import settings
from schemas.core import Event
from store.event_store import EventStore
from store.trace_ledger import TraceEntry, TraceLedger

T = TypeVar("T", bound=BaseModel)


class UnifiedObserver:
    def __init__(self, event_store: EventStore | None = None, trace_ledger: TraceLedger | None = None) -> None:
        self.event_store = event_store
        self.trace_ledger = trace_ledger
        self.client = None
        if settings.llm_api_key:
            base_client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
            self.client = instructor.patch(base_client, mode=instructor.Mode.JSON)

    def record_events(self, events: Iterable[Event]) -> None:
        if self.event_store is not None:
            self.event_store.append(events)

    def record_trace(self, entry: TraceEntry) -> None:
        if self.trace_ledger is not None:
            self.trace_ledger.append(entry)

    def perceive(self, context: dict[str, Any], response_model: type[T]) -> T:
        items = context.get("items")
        ranked_ids: list[str] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    item_id = item.get("id")
                    if isinstance(item_id, str):
                        ranked_ids.append(item_id)
        return response_model(ranked_ids=ranked_ids)


class NullObserver:
    def record_events(self, events: Iterable[Event]) -> None:  # noqa: ARG002
        return None

    def record_trace(self, entry: TraceEntry) -> None:  # noqa: ARG002
        return None
