from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from sqlmodel import Field, Session, SQLModel, create_engine, select

from config.settings import settings
from schemas.memory import MemoryItem, MemoryStats


class MemoryRow(SQLModel, table=True):
    id: str = Field(primary_key=True)
    kind: str
    pointer_json: str
    snippet: str
    dimensions_json: str
    stats_json: str


class MemoryStore:
    """SQLite-backed store for structured memory items.

    Notes:
    - ``dimensions`` and ``stats`` must be JSON-serializable dictionaries (e.g., may include ``created_at``).
    - ``stats.recent_activity_score`` is a simple proxy currently based on total row count.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.memory_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self.engine)

    def upsert_item(self, item: MemoryItem) -> None:
        try:
            pointer_json = json.dumps(item.pointer)
            dimensions_json = json.dumps(item.dimensions)
            stats_json = json.dumps(item.stats)
        except (TypeError, ValueError) as exc:
            raise ValueError("MemoryStore items must be JSON-serializable") from exc

        with Session(self.engine) as session:
            existing = session.get(MemoryRow, item.id)
            payload = dict(
                id=item.id,
                kind=item.kind,
                pointer_json=pointer_json,
                snippet=item.snippet,
                dimensions_json=dimensions_json,
                stats_json=stats_json,
            )
            if existing:
                for key, value in payload.items():
                    setattr(existing, key, value)
                session.add(existing)
            else:
                session.add(MemoryRow(**payload))
            session.commit()

    def get_item(self, item_id: str) -> MemoryItem | None:
        with Session(self.engine) as session:
            row = session.get(MemoryRow, item_id)
            if not row:
                return None
        return MemoryItem(
            id=row.id,
            kind=row.kind,  # type: ignore[arg-type]
            pointer=json.loads(row.pointer_json),
            snippet=row.snippet,
            dimensions=json.loads(row.dimensions_json),
            stats=json.loads(row.stats_json),
        )

    def query_by_dimensions(self, filters: dict[str, Any], limit: int = 100) -> list[MemoryItem]:
        with Session(self.engine) as session:
            statement = select(MemoryRow).limit(limit)
            rows = session.exec(statement).all()
        results: list[MemoryItem] = []
        for row in rows:
            try:
                dimensions = json.loads(row.dimensions_json)
            except (TypeError, ValueError):
                continue
            if not isinstance(dimensions, dict):
                continue
            if all(str(dimensions.get(k)) == str(v) for k, v in filters.items()):
                try:
                    pointer = json.loads(row.pointer_json)
                    stats = json.loads(row.stats_json)
                except (TypeError, ValueError):
                    continue
                results.append(
                    MemoryItem(
                        id=row.id,
                        kind=row.kind,  # type: ignore[arg-type]
                        pointer=pointer,
                        snippet=row.snippet,
                        dimensions=dimensions,
                        stats=stats,
                    )
                )
        return results

    def stats(self) -> MemoryStats:
        with Session(self.engine) as session:
            statement = select(MemoryRow.kind)
            kinds = session.exec(statement).all()
        counts: dict[str, int] = {}
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
        recent_activity_score = float(len(kinds))
        return MemoryStats(counts_by_kind=counts, recent_activity_score=recent_activity_score)
