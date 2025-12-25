from __future__ import annotations

from typing import Iterable, Type

import instructor
from openai import OpenAI
from pydantic import BaseModel

from config.settings import settings
from schemas.core import Event
from store.event_store import EventStore
from store.trace_ledger import TraceEntry, TraceLedger


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

    def perceive(
        self,
        instruction: str,
        response_model: Type[BaseModel],
        context_text: str = "",
        temperature: float = 0.0,
    ) -> BaseModel:
        if self.client is None:
            raise RuntimeError("Observer LLM client is not configured; set settings.llm_api_key to use perceive().")

        messages = [{"role": "system", "content": instruction}]
        if context_text:
            messages.append({"role": "user", "content": context_text})

        return self.client.chat.completions.create(
            model=settings.llm_model_observer,
            messages=messages,
            temperature=temperature,
            response_model=response_model,
        )


class NullObserver:
    def record_events(self, events: Iterable[Event]) -> None:  # noqa: ARG002
        return None

    def record_trace(self, entry: TraceEntry) -> None:  # noqa: ARG002
        return None
