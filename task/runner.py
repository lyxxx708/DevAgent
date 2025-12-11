from __future__ import annotations

import uuid
from typing import Dict, List

from meta.controller import MetaController
from schemas.core import Event, Program, State
from schemas.views import AgentHints, DecisionInputView, GoalView


class TaskRunner:
    """In-memory task lifecycle manager that delegates execution to MetaController."""

    def __init__(self, controller: MetaController) -> None:
        self.controller = controller
        self._states: Dict[str, State] = {}
        self._goals: Dict[str, GoalView] = {}
        self._step_ids: Dict[str, int] = {}

    def create_job(self, repo_root: str, goal_view: GoalView) -> str:
        job_id = str(uuid.uuid4())
        state = State(git_head="", repo_root=repo_root)
        self._states[job_id] = state
        self._goals[job_id] = goal_view
        self._step_ids[job_id] = 0
        return job_id

    def run_step(
        self,
        job_id: str,
        program: Program,
        *,
        hints: AgentHints | None = None,
    ) -> tuple[State, List[Event], DecisionInputView]:
        """Execute one MetaController-coordinated step for an existing job, advancing state and step counters."""
        if job_id not in self._states:
            raise ValueError(f"Unknown job_id: {job_id}")

        state = self._states[job_id]
        goal_view = self._goals[job_id]
        start_step_id = self._step_ids[job_id] + 1

        new_state, events, decision_input = self.controller.run_step(
            job_id=job_id,
            state=state,
            program=program,
            goal_view=goal_view,
            hints=hints,
            start_step_id=start_step_id,
        )

        self._states[job_id] = new_state
        if events:
            max_step = max(event.step_id for event in events)
            self._step_ids[job_id] = max_step
        return new_state, events, decision_input
