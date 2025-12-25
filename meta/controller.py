from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agent.devagent import DevAgent
from memory.store import MemoryStore
from meta.llm_meta_planner import LLMMetaPlanner
from schemas.core import Event, Program, State
from schemas.meta import FocusSpec, GoalViewSummary, MetaInputView, RerankHints, SelectorProfile, StateSummary
from schemas.views import AgentHints, DecisionInputView, DevAgentMode, GoalView, MemoryView, StateView
from store.event_store import EventStore
from store.trace_ledger import TraceEntry, TraceLedger
from views.focus import LLMFocusInferer


class MetaController:
    """Meta-level orchestrator that wraps DevAgent with planning and tracing.

    It builds a MetaInputView from goal context, a lightweight StateSummary derived
    from recent events, and MemoryStats, then delegates execution to DevAgent while
    recording trace entries.
    """

    def __init__(
        self,
        devagent: DevAgent,
        planner: LLMMetaPlanner,
        memory_store: MemoryStore,
        trace_ledger: TraceLedger,
        event_store: EventStore,
    ) -> None:
        self.devagent = devagent
        self.planner = planner
        self.memory_store = memory_store
        self.trace_ledger = trace_ledger
        self.event_store = event_store
        self.llm_focus_inferer = LLMFocusInferer()

    def _get_recent_error_logs(self, job_id: str, limit: int = 3, max_chars: int = 2000) -> str:
        recent = self.event_store.recent_for_job(job_id, limit=200)
        failures: list[str] = []
        for event in recent:
            if event.type != "RUN":
                continue
            payload = event.payload or {}
            exit_code = int(payload.get("exit_code", 0) or 0)
            if exit_code == 0:
                continue
            stderr = str(payload.get("stderr", "") or "")
            if max_chars and len(stderr) > max_chars:
                stderr = stderr[:max_chars]
            cmd = payload.get("cmd")
            header = f"RUN failed (exit_code={exit_code})"
            if isinstance(cmd, str):
                header = f"{header}: {cmd}"
            failures.append(f"{header}\n{stderr}")
            if len(failures) >= limit:
                break
        return "\n\n---\n\n".join(failures)

    def _get_repo_tree(self, repo_root: str, max_depth: int = 2) -> str:
        root = Path(repo_root).resolve()
        if not root.exists():
            return ""
        lines: list[str] = [f"{root.name}/"]
        for current_path in sorted(root.iterdir()):
            self._append_tree_lines(lines, root, current_path, depth=1, max_depth=max_depth)
        return "\n".join(lines)

    def _append_tree_lines(
        self,
        lines: list[str],
        root: Path,
        path: Path,
        *,
        depth: int,
        max_depth: int,
    ) -> None:
        if depth > max_depth:
            return
        try:
            rel_path = path.relative_to(root)
        except ValueError:
            return
        if rel_path.parts and rel_path.parts[0] == ".git":
            return
        indent = "  " * depth
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{indent}- {path.name}{suffix}")
        if path.is_dir() and depth < max_depth:
            for child in sorted(path.iterdir()):
                self._append_tree_lines(lines, root, child, depth=depth + 1, max_depth=max_depth)

    def _read_focus_file_contents(
        self,
        repo_root: str,
        focus_files: list[str],
        max_chars: int = 4000,
    ) -> str:
        root = Path(repo_root).resolve()
        snippets: list[str] = []
        for file_path in focus_files:
            candidate = (root / file_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if not candidate.is_file():
                continue
            try:
                content = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if max_chars and len(content) > max_chars:
                content = content[:max_chars]
            snippets.append(f"File: {file_path}\n{content}")
        return "\n\n".join(snippets)

    def _build_bootstrap_prompt(
        self,
        goal_view: GoalView,
        focus_files_content: str,
        repo_tree: str,
        recent_errors: str,
        memory_view: MemoryView,
    ) -> str:
        memory_lines = []
        for item in memory_view.items[:10]:
            snippet = item.snippet or ""
            memory_lines.append(f"- ({item.kind}) {snippet}")
        memory_block = "\n".join(memory_lines)
        sections = [
            f"Goal: {goal_view.natural_language_goal}",
            "Repo tree:",
            repo_tree or "(empty)",
            "Recent failures:",
            recent_errors or "(none)",
            "Memory highlights:",
            memory_block or "(none)",
            "Focus file contents:",
            focus_files_content or "(none)",
        ]
        return "\n\n".join(sections)

    def _run_bootstrap_llm_heavy(
        self,
        job_id: str,
        state: State,
        goal_view: GoalView,
        *,
        hints: AgentHints | None,
        focus_spec: FocusSpec,
        selector_profile: SelectorProfile,
        rerank_hints: RerankHints | None,
        start_step_id: int,
    ) -> tuple[State, list[Event], DecisionInputView]:
        repo_tree = self._get_repo_tree(state.repo_root)
        recent_errors = self._get_recent_error_logs(job_id)

        try:
            focus_view = self.llm_focus_inferer.infer(repo_tree, recent_errors, focus_spec)
        except NotImplementedError:
            focus_view = self.devagent.baseline_focus_inferer.infer([])

        memory_candidates = self.devagent.selector.select(
            profile=selector_profile,
            filters=None,
            limit=50,
        )
        reranked_items = self.devagent.reranker.rerank(memory_candidates, hints=rerank_hints)
        memory_view = MemoryView(items=reranked_items, stats=self.memory_store.stats())

        focus_files_content = self._read_focus_file_contents(state.repo_root, focus_view.files)
        prompt = self._build_bootstrap_prompt(
            goal_view=goal_view,
            focus_files_content=focus_files_content,
            repo_tree=repo_tree,
            recent_errors=recent_errors,
            memory_view=memory_view,
        )

        decision_input = DecisionInputView(
            state_view=StateView(
                git_head=state.git_head,
                failing_tests=[],
                repo_stats={},
            ),
            focus_view=focus_view,
            memory_view=memory_view,
            goal_view=goal_view,
            hints=hints or AgentHints(),
            mode=self.devagent.mode,
            token_budget_hint=None,
        )

        program = self.devagent.devagent_step(decision_input, prompt)
        new_state, events = self.devagent.execute_program(
            state=state,
            program=program,
            job_id=job_id,
            start_step_id=start_step_id,
        )
        return new_state, events, decision_input

    def _build_state_summary(self, job_id: str) -> StateSummary:
        """Derive a lightweight StateSummary from recent events."""

        recent = self.event_store.recent_for_job(job_id, limit=200)
        failing_tests_count = 0
        for evt in recent:
            if evt.type != "RUN":
                continue
            payload = evt.payload or {}
            exit_code = int(payload.get("exit_code", 0) or 0)
            if exit_code != 0:
                failing_tests_count += 1

        return StateSummary(repo_size=0, failing_tests_count=failing_tests_count, key_modules=[])

    def _mode_literal(self, mode: DevAgentMode) -> str:
        if mode == DevAgentMode.BOOTSTRAP_LLM_HEAVY:
            return "bootstrap_llm_heavy"
        return "optimized_structured"

    def run_step(
        self,
        job_id: str,
        state: State,
        program: Program,
        goal_view: GoalView,
        *,
        hints: AgentHints | None = None,
        start_step_id: int = 1,
    ) -> tuple[State, list[Event], DecisionInputView]:
        """Plan and execute a single step: build MetaInputView, call planner, delegate to DevAgent, and trace the outcome."""
        goal_summary = GoalViewSummary(
            task_type=goal_view.task_type,
            natural_language_goal=goal_view.natural_language_goal,
        )
        state_summary = self._build_state_summary(job_id)
        memory_stats = self.memory_store.stats()

        meta_input = MetaInputView(
            goal_view=goal_summary,
            state_summary=state_summary,
            memory_stats=memory_stats,
            trace_hint=None,
            mode=self._mode_literal(self.devagent.mode),
        )

        plan = self.planner.propose_plan(meta_input)
        if self.devagent.mode == DevAgentMode.BOOTSTRAP_LLM_HEAVY:
            new_state, events, decision_input = self._run_bootstrap_llm_heavy(
                job_id=job_id,
                state=state,
                goal_view=goal_view,
                hints=hints,
                focus_spec=plan.focus_spec,
                selector_profile=plan.selector_profile,
                rerank_hints=plan.rerank_hints,
                start_step_id=start_step_id,
            )
        else:
            new_state, events, decision_input = self.devagent.run_step(
                job_id=job_id,
                state=state,
                program=program,
                goal_view=goal_view,
                hints=hints,
                focus_spec=plan.focus_spec,
                selector_profile=plan.selector_profile,
                start_step_id=start_step_id,
                rerank_hints=plan.rerank_hints,
            )

        decision_id = str(uuid.uuid4())
        step_id = max((event.step_id for event in events), default=start_step_id)
        decision_input_summary: dict[str, Any] = {
            "goal_task_type": decision_input.goal_view.task_type,
            "files": decision_input.focus_view.files,
            "mode": decision_input.mode,
        }
        program_summary = {"instruction_count": len(program.instructions)}
        outcome_summary = {
            "event_count": len(events),
            "git_head": new_state.git_head,
            "rerank_hints": plan.rerank_hints.model_dump() if plan.rerank_hints else None,
        }

        entry = TraceEntry(
            decision_id=decision_id,
            job_id=job_id,
            step_id=step_id,
            decision_input_summary=decision_input_summary,
            program_summary=program_summary,
            outcome_summary=outcome_summary,
        )
        self.trace_ledger.append(entry)

        return new_state, events, decision_input
