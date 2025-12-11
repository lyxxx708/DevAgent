from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select

from config.settings import settings


class TraceEntry(BaseModel):
    decision_id: str
    job_id: str
    step_id: int
    decision_input_summary: dict[str, Any]
    program_summary: dict[str, Any]
    outcome_summary: dict[str, Any]


class TraceRow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    decision_id: str
    job_id: str
    step_id: int
    payload_json: str


class TraceLedger:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.trace_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self.engine)

    def append(self, entry: TraceEntry) -> None:
        payload_json = json.dumps(entry.model_dump())
        with Session(self.engine) as session:
            row = TraceRow(
                decision_id=entry.decision_id,
                job_id=entry.job_id,
                step_id=entry.step_id,
                payload_json=payload_json,
            )
            session.add(row)
            session.commit()

    def recent_for_job(self, job_id: str, limit: int = 100) -> list[TraceEntry]:
        with Session(self.engine) as session:
            statement = (
                select(TraceRow)
                .where(TraceRow.job_id == job_id)
                .order_by(TraceRow.step_id.desc(), TraceRow.id.desc())
                .limit(limit)
            )
            rows = session.exec(statement).all()
        entries: list[TraceEntry] = []
        for row in rows:
            data = json.loads(row.payload_json)
            entries.append(TraceEntry(**data))
        return entries
