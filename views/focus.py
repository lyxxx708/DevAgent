from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel

from infra.observer import UnifiedObserver
from memory.selector import MemorySelector
from memory.reranker import MemoryReranker
from schemas.core import Event
from schemas.meta import FocusSpec, RerankHints, SelectorProfile
from schemas.views import FocusView


class BaselineFocusInferer:
    """Non-LLM focus inference based on failing RUN events."""

    SOURCE_PATTERN = re.compile(r"(src/[^\s:]+\.py)")
    TEST_PATTERN = re.compile(r"(tests/[^\s:]+\.py)")

    def infer(self, events: Iterable[Event]) -> FocusView:
        source_files: set[str] = set()
        test_files: set[str] = set()
        for event in events:
            if event.type != "RUN":
                continue
            exit_code = event.payload.get("exit_code", 0) or 0
            if exit_code == 0:
                continue
            stderr = event.payload.get("stderr", "") or ""
            for match in self.SOURCE_PATTERN.findall(stderr):
                source_files.add(match)
            for match in self.TEST_PATTERN.findall(stderr):
                test_files.add(match)
        return FocusView(files=list(source_files), modules=[], tests=list(test_files))


class LLMFocusInferer:
    """LLM-based focus inferrer that delegates to a UnifiedObserver."""

    def __init__(self, observer: UnifiedObserver, baseline: BaselineFocusInferer | None = None) -> None:
        self.observer = observer
        self.baseline = baseline

    def infer(
        self,
        file_tree: str,
        failing_logs: str,
        focus_spec: FocusSpec | None = None,
        *,
        events: Iterable[Event] | None = None,
    ) -> FocusView:
        if focus_spec is None:
            focus_spec = FocusSpec(task_type="unknown", modules=[], only_failing_tests=True, max_focus_files=20)

        context_parts = []
        if file_tree.strip():
            context_parts.append(f"File tree:\n{file_tree.strip()}")
        if failing_logs.strip():
            context_parts.append(f"Failing logs:\n{failing_logs.strip()}")
        context = "\n\n".join(context_parts).strip()

        task_description = (
            "Review the repository file tree and failing test logs to identify likely culprit files and relevant tests. "
            "Return reasoning, culprit_files, and relevant_tests."
        )

        try:
            response = self.observer.perceive(
                task=task_description,
                context=context,
                response_model=FocusResult,
            )
            if isinstance(response, FocusResult):
                result = response
            elif isinstance(response, dict):
                result = FocusResult(**response)
            else:
                raise TypeError("Unsupported response type from observer.perceive")

            files = list(dict.fromkeys(result.culprit_files))
            tests = list(dict.fromkeys(result.relevant_tests))
            if focus_spec.max_focus_files is not None:
                files = files[: focus_spec.max_focus_files]
            return FocusView(files=files, modules=[], tests=tests)
        except Exception:
            if self.baseline is not None and events is not None:
                fallback_view = self.baseline.infer(events)
                files = fallback_view.files
                if focus_spec.max_focus_files is not None:
                    files = files[: focus_spec.max_focus_files]
                return FocusView(files=files, modules=fallback_view.modules, tests=fallback_view.tests)
            return FocusView(files=[], modules=[], tests=[])


class FocusResult(BaseModel):
    reasoning: str
    culprit_files: list[str]
    relevant_tests: list[str]


class FocusViewBuilder:
    """Builds FocusView from selector/reranker outputs.

    Rerank hints may adjust ordering via the MemoryReranker before files/modules/tests are derived. The current
    implementation derives files/modules/tests from item dimensions (e.g., ``file_path``/``module``/``modules``/``test_path``),
    truncates files using ``spec.max_focus_files``, and currently ignores ``spec.only_failing_tests`` and ``spec.modules``.
    """

    def __init__(self, selector: MemorySelector, reranker: MemoryReranker) -> None:
        self.selector = selector
        self.reranker = reranker

    def build(
        self,
        spec: FocusSpec,
        profile: SelectorProfile,
        filters: dict[str, Any] | None = None,
        hints: RerankHints | None = None,
    ) -> FocusView:
        candidates = self.selector.select(profile=profile, filters=filters)
        reranked = self.reranker.rerank(candidates, hints=hints)

        files: list[str] = []
        modules: list[str] = []
        tests: list[str] = []

        for item in reranked:
            file_path = item.dimensions.get("file_path") if isinstance(item.dimensions, dict) else None
            if isinstance(file_path, str) and file_path:
                files.append(file_path)
            module_value = item.dimensions.get("module") if isinstance(item.dimensions, dict) else None
            if isinstance(module_value, str) and module_value:
                modules.append(module_value)
            modules_value = item.dimensions.get("modules") if isinstance(item.dimensions, dict) else None
            if isinstance(modules_value, list):
                modules.extend([m for m in modules_value if isinstance(m, str)])
            test_path = item.dimensions.get("test_path") if isinstance(item.dimensions, dict) else None
            if isinstance(test_path, str) and test_path:
                tests.append(test_path)

        files = list(dict.fromkeys(files))
        modules = list(dict.fromkeys(modules))
        tests = list(dict.fromkeys(tests))
        if spec.max_focus_files is not None:
            files = files[: spec.max_focus_files]

        return FocusView(files=files, modules=modules, tests=tests)
