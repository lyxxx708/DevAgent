from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.devagent import DevAgent
from config.settings import Settings, settings as global_settings
from infra.observer import UnifiedObserver
from infra.vector_store import VectorStore
from memory.store import MemoryStore
from meta.controller import MetaController
from meta.llm_meta_planner import LLMMetaPlanner
from schemas.core import Event, Program, State
from schemas.views import AgentHints, DecisionInputView, DevAgentMode, GoalView
from store.event_store import EventStore
from store.trace_ledger import TraceLedger
from task.runner import TaskRunner


# Fixed detail string for missing jobs to keep HTTP responses deterministic.
JOB_NOT_FOUND_DETAIL = "job not found"


class CreateJobRequest(BaseModel):
    """Request model for creating a DevAgent job tied to a repo root and goal."""

    repo_root: str
    goal: GoalView


class CreateJobResponse(BaseModel):
    """Response containing the allocated job identifier."""

    job_id: str


class RunStepRequest(BaseModel):
    """Request model for executing a single step of a job using a Program and optional hints."""

    program: Program
    hints: AgentHints | None = None


class RunStepResponse(BaseModel):
    """Response model returning state, events, and decision context from a run step."""

    job_id: str
    state: State
    events: list[Event]
    decision: DecisionInputView


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a FastAPI app wired to the DevAgent v7.3 stack (stores, planner, controller, task runner)."""
    app_settings = settings or global_settings

    app = FastAPI()

    event_store = EventStore(db_path=app_settings.event_db_path)
    trace_ledger = TraceLedger(db_path=app_settings.trace_db_path)
    memory_store = MemoryStore(db_path=app_settings.memory_db_path)
    observer = UnifiedObserver(event_store=event_store, trace_ledger=trace_ledger)
    vector_store = VectorStore(dim=app_settings.vector_dim, use_faiss=False)
    devagent = DevAgent(
        mode=DevAgentMode.OPTIMIZED_STRUCTURED,
        observer=observer,
        memory_store=memory_store,
        vector_store=vector_store,
    )
    planner = LLMMetaPlanner()
    meta_controller = MetaController(
        devagent=devagent,
        planner=planner,
        memory_store=memory_store,
        trace_ledger=trace_ledger,
    )
    task_runner = TaskRunner(controller=meta_controller)

    app.state.event_store = event_store
    app.state.trace_ledger = trace_ledger
    app.state.memory_store = memory_store
    app.state.vector_store = vector_store
    app.state.devagent = devagent
    app.state.meta_controller = meta_controller
    app.state.task_runner = task_runner

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "mode": devagent.mode.value}

    @app.post("/jobs", response_model=CreateJobResponse)
    def create_job(request: CreateJobRequest) -> CreateJobResponse:
        job_id = task_runner.create_job(repo_root=request.repo_root, goal_view=request.goal)
        return CreateJobResponse(job_id=job_id)

    @app.post("/jobs/{job_id}/steps", response_model=RunStepResponse)
    def run_step(job_id: str, request: RunStepRequest) -> RunStepResponse:
        try:
            state, events, decision = task_runner.run_step(
                job_id=job_id,
                program=request.program,
                hints=request.hints,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=JOB_NOT_FOUND_DETAIL) from exc
        return RunStepResponse(job_id=job_id, state=state, events=events, decision=decision)

    return app


app = create_app()
