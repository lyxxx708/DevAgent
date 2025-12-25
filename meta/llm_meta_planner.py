from __future__ import annotations

"""Baseline, non-LLM meta planner for DevAgent v7.3.

This heuristic planner is intentionally lightweight and contains no network calls
or LLM dependencies. It maps a MetaInputView into a MetaPlan by choosing focus
and selection policies based on the goal type and simple repository signals.
"""

import time
from typing import Any, Dict

from pydantic import BaseModel, ValidationError

from infra.observer import UnifiedObserver
from schemas.meta import FocusSpec, MetaInputView, MetaPlan, RerankHints, SelectorProfile


class MetaPlanResult(BaseModel):
    focus_spec: FocusSpec
    selector_profile: SelectorProfile
    rerank_hints: RerankHints | None = None


class LLMMetaPlanner:
    """Rule-based placeholder for the future LLM-backed meta planner.

    The policy here is deliberately simple and fully deterministic so it can run
    without any external services. It aligns with the schemas defined in
    ``schemas.meta`` and can be replaced later by an LLM-backed planner without
    changing the public API. This planner is heuristic-only, performs zero
    network/LLM calls, and biases error_pattern memories when failures are
    present.
    """

    def __init__(self, observer: UnifiedObserver | None = None) -> None:
        self.observer = observer

    def propose_plan(self, meta_input: MetaInputView) -> MetaPlan:
        payload = self._build_perception_payload(meta_input)
        result = self._request_llm_plan(payload)
        if result is not None:
            return MetaPlan(
                focus_spec=result.focus_spec,
                selector_profile=result.selector_profile,
                rerank_hints=result.rerank_hints,
            )
        return self._fallback_plan(meta_input)

    def _build_perception_payload(self, meta_input: MetaInputView) -> dict[str, Any]:
        return {
            "kind": "meta_plan_request",
            "meta_input": meta_input.model_dump(),
        }

    def _request_llm_plan(self, payload: dict[str, Any]) -> MetaPlanResult | None:
        if self.observer is None:
            return None
        response = self.observer.perceive(payload)
        if response is None:
            return None
        try:
            return MetaPlanResult.model_validate(response)
        except ValidationError:
            return None

    def _fallback_plan(self, meta_input: MetaInputView) -> MetaPlan:
        goal_type = meta_input.goal_view.task_type

        only_failing_tests = goal_type == "fix_failures"
        if goal_type == "fix_failures":
            max_focus_files = 20
        elif goal_type == "init_project":
            max_focus_files = 10
        else:
            max_focus_files = 15

        focus_spec = FocusSpec(
            task_type=goal_type,
            modules=[],
            only_failing_tests=only_failing_tests,
            max_focus_files=max_focus_files,
        )

        per_kind_limit: Dict[str, int] = {"error_pattern": 10, "run_config": 10}
        weights: Dict[str, float] = {}
        if meta_input.state_summary.failing_tests_count > 0:
            weights["error_pattern"] = 1.0

        recency_window = None
        if meta_input.memory_stats.recent_activity_score > 5:
            recency_window = int(time.time()) - 3600

        selector_profile = SelectorProfile(
            weights=weights,
            per_kind_limit=per_kind_limit,
            recency_window=recency_window,
        )

        rerank_hints = RerankHints(
            boost_dimensions={"layer": 1.0} if meta_input.mode == "bootstrap_llm_heavy" else {},
            diversity_over=None,
            prefer_recent=True,
        )

        return MetaPlan(
            focus_spec=focus_spec,
            selector_profile=selector_profile,
            rerank_hints=rerank_hints,
        )


__all__ = ["LLMMetaPlanner", "MetaPlanResult"]
