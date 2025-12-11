from __future__ import annotations

from typing import Iterable
import time

from loguru import logger

from schemas.core import Event
from schemas.memory import MemoryItem
from memory.store import MemoryStore


class MemoryIngestPipeline:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def ingest(self, events: Iterable[Event]) -> None:
        for event in events:
            try:
                if event.type == "RUN":
                    exit_code = event.payload.get("exit_code", 0) or 0
                    stderr = event.payload.get("stderr", "") or ""
                    snippet_tail = stderr[:200]
                    pointer = {"event_id": event.event_id, "job_id": event.job_id}
                    dimensions = {"kind": "run", "exit_code": exit_code, "job_id": event.job_id}
                    stats = {"created_at": time.time(), "access_count": 0}
                    if exit_code != 0:
                        kind = "error_pattern"
                        snippet = f"RUN failed with exit_code={exit_code}, stderr={snippet_tail}"
                    else:
                        kind = "run_config"
                        snippet = f"RUN succeeded with exit_code={exit_code}"
                    item = MemoryItem(
                        id=event.event_id,
                        kind=kind,  # type: ignore[arg-type]
                        pointer=pointer,
                        snippet=snippet,
                        dimensions=dimensions,
                        stats=stats,
                    )
                    self.store.upsert_item(item)
                elif event.type == "EDIT":
                    file_path = event.payload.get("file_path")
                    pointer = {"event_id": event.event_id, "job_id": event.job_id, "file_path": file_path}
                    dimensions = {"kind": "edit", "file_path": file_path}
                    stats = {"created_at": time.time(), "access_count": 0}
                    snippet = f"Edited file: {file_path}" if file_path else "Edited file"
                    item = MemoryItem(
                        id=event.event_id,
                        kind="module_history",  # type: ignore[arg-type]
                        pointer=pointer,
                        snippet=snippet,
                        dimensions=dimensions,
                        stats=stats,
                    )
                    self.store.upsert_item(item)
            except Exception:
                logger.exception("Failed to ingest event", event_id=event.event_id)
