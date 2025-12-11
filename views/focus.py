from __future__ import annotations

import re
from typing import Any, Iterable

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
    """Placeholder for a future LLM-based focus inferrer."""

    def __init__(self) -> None:
        pass

    def infer(self, file_tree: str, failing_logs: str, focus_spec: FocusSpec | None = None) -> FocusView:
        raise NotImplementedError("LLMFocusInferer is not implemented yet in this environment.")


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
