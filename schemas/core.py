from typing import Any, Literal

from pydantic import BaseModel


class StateDiagnostics(BaseModel):
    memory_mode: Literal["OK", "DEGRADED_PARTIAL", "DOWN"] = "OK"
    last_error: str | None = None


class State(BaseModel):
    git_head: str
    repo_root: str
    config_profile: str = "default"
    diagnostics: StateDiagnostics = StateDiagnostics()


class Instruction(BaseModel):
    kind: Literal["RUN", "EDIT", "META"]
    payload: dict[str, Any]


class Program(BaseModel):
    instructions: list[Instruction]


class Event(BaseModel):
    event_id: str
    job_id: str
    step_id: int
    type: Literal["RUN", "EDIT", "META", "SYSTEM"]
    payload: dict[str, Any]
    started_at: float
    ended_at: float


__all__ = [
    "StateDiagnostics",
    "State",
    "Instruction",
    "Program",
    "Event",
]
