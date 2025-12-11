from __future__ import annotations

import os
import tempfile

import pytest

from memory.store import MemoryStore
from schemas.memory import MemoryItem


def test_query_filters_and_stats_are_sane() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(db_path=os.path.join(tmpdir, "memory.db"))
        item_a = MemoryItem(
            id="a",
            kind="run_config",
            pointer={},
            snippet="",
            dimensions={"file_path": "src/a.py", "layer": "L1"},
            stats={"created_at": 1.0},
        )
        item_b = MemoryItem(
            id="b",
            kind="error_pattern",
            pointer={},
            snippet="",
            dimensions={"file_path": "src/b.py", "layer": "L2"},
            stats={"created_at": 2.0},
        )
        store.upsert_item(item_a)
        store.upsert_item(item_b)

        filtered = store.query_by_dimensions({"file_path": "src/a.py"})
        assert len(filtered) == 1
        assert filtered[0].id == "a"

        stats = store.stats()
        assert stats.recent_activity_score >= 0
        assert sum(stats.counts_by_kind.values()) == 2


def test_upsert_requires_json_serializable_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(db_path=os.path.join(tmpdir, "memory.db"))
        bad_item = MemoryItem(
            id="bad",
            kind="run_config",
            pointer={},
            snippet="",
            dimensions={"invalid": set([1])},
            stats={},
        )

        with pytest.raises(ValueError):
            store.upsert_item(bad_item)
