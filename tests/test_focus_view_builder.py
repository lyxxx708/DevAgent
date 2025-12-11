from __future__ import annotations

import os
import tempfile

from memory.selector import MemorySelector
from memory.reranker import MemoryReranker
from memory.store import MemoryStore
from schemas.memory import MemoryItem
from schemas.meta import FocusSpec, SelectorProfile
from views.focus import FocusViewBuilder


def test_focus_view_builder_collects_files_with_limit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "memory.db")
        store = MemoryStore(db_path=store_path)
        selector = MemorySelector(store=store)
        reranker = MemoryReranker()
        builder = FocusViewBuilder(selector=selector, reranker=reranker)

        items = [
            MemoryItem(
                id="m1",
                kind="run_config",
                pointer={},
                snippet="",
                dimensions={"file_path": "src/a.py"},
                stats={},
            ),
            MemoryItem(
                id="m2",
                kind="run_config",
                pointer={},
                snippet="",
                dimensions={"file_path": "src/b.py"},
                stats={},
            ),
            MemoryItem(
                id="m3",
                kind="run_config",
                pointer={},
                snippet="",
                dimensions={"file_path": "src/c.py"},
                stats={},
            ),
        ]
        for item in items:
            store.upsert_item(item)

        spec = FocusSpec(task_type="fix_failures", modules=[], only_failing_tests=True, max_focus_files=2)
        profile = SelectorProfile(weights={}, per_kind_limit={}, recency_window=None)

        focus_view = builder.build(spec=spec, profile=profile)
        assert len(focus_view.files) <= 2
        assert set(focus_view.files).issubset({"src/a.py", "src/b.py", "src/c.py"})
