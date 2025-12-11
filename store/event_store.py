from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

from sqlmodel import Field, Session, SQLModel, create_engine, select

from config.settings import settings
from schemas.core import Event


class EventRow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: str
    step_id: int
    event_id: str
    type: str
    payload_json: str
    started_at: float
    ended_at: float


class EventStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.event_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self.engine)

    def append(self, events: Iterable[Event]) -> None:
        with Session(self.engine) as session:
            for event in events:
                row = EventRow(
                    job_id=event.job_id,
                    step_id=event.step_id,
                    event_id=event.event_id,
                    type=event.type,
                    payload_json=json.dumps(event.payload),
                    started_at=event.started_at,
                    ended_at=event.ended_at,
                )
                session.add(row)
            session.commit()

    def recent_for_job(self, job_id: str, limit: int = 200) -> list[Event]:
        with Session(self.engine) as session:
            statement = (
                select(EventRow)
                .where(EventRow.job_id == job_id)
                .order_by(EventRow.step_id.desc(), EventRow.id.desc())
                .limit(limit)
            )
            rows = session.exec(statement).all()
        events: list[Event] = []
        for row in rows:
            events.append(
                Event(
                    event_id=row.event_id,
                    job_id=row.job_id,
                    step_id=row.step_id,
                    type=row.type,  # type: ignore[arg-type]
                    payload=json.loads(row.payload_json),
                    started_at=row.started_at,
                    ended_at=row.ended_at,
                )
            )
        return events
