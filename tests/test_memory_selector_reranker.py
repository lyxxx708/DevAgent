from __future__ import annotations

import os
import tempfile

from memory.selector import MemorySelector
from memory.reranker import MemoryReranker
from memory.store import MemoryStore
from schemas.memory import MemoryItem
from schemas.meta import RerankHints, SelectorProfile


def _make_item(item_id: str, kind: str, file_path: str, created_at: float) -> MemoryItem:
    return MemoryItem(
        id=item_id,
        kind=kind,
        pointer={},
        snippet="",  # noqa: P103
        dimensions={"file_path": file_path},
        stats={"created_at": created_at},
    )


def test_selector_respects_per_kind_limits_and_vector_order() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "memory.db")
        store = MemoryStore(db_path=store_path)
        selector = MemorySelector(store=store, vector_store=None)

        items = [
            _make_item("i1", "error_pattern", "src/a.py", 1.0),
            _make_item("i2", "error_pattern", "src/b.py", 2.0),
            _make_item("i3", "run_config", "src/c.py", 3.0),
        ]
        for item in items:
            store.upsert_item(item)

        profile = SelectorProfile(weights={}, per_kind_limit={"error_pattern": 1, "run_config": 2}, recency_window=None)
        selected = selector.select(profile=profile, filters={})
        assert len(selected) == 2
        assert sum(1 for i in selected if i.kind == "error_pattern") <= 1


def test_selector_with_vector_prioritizes_closest() -> None:
    from infra.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "memory.db")
        store = MemoryStore(db_path=store_path)
        vector_store = VectorStore(dim=2, use_faiss=False)
        selector = MemorySelector(store=store, vector_store=vector_store)

        items = [
            _make_item("close", "run_config", "src/close.py", 1.0),
            _make_item("far", "run_config", "src/far.py", 1.0),
        ]
        vectors = {
            "close": [0.0, 0.0],
            "far": [5.0, 5.0],
        }
        for item in items:
            store.upsert_item(item)
        vector_store.add(list(vectors.keys()), list(vectors.values()))

    profile = SelectorProfile(weights={}, per_kind_limit={}, recency_window=None)
    selected = selector.select(profile=profile, filters={}, query_vector=[0.1, 0.1])
    assert selected[0].id == "close"


def test_selector_vector_empty_results_keep_candidates() -> None:
    from infra.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "memory.db")
        store = MemoryStore(db_path=store_path)
        vector_store = VectorStore(dim=2, use_faiss=False)
        selector = MemorySelector(store=store, vector_store=vector_store)

        items = [
            _make_item("a", "run_config", "src/a.py", 1.0),
            _make_item("b", "run_config", "src/b.py", 1.0),
        ]
        for item in items:
            store.upsert_item(item)

        profile = SelectorProfile(weights={}, per_kind_limit={}, recency_window=None)
        selected = selector.select(profile=profile, filters={}, query_vector=[0.0, 0.0])

        assert len(selected) == 2


def test_reranker_prefers_recent_and_boosts_dimensions() -> None:
    reranker = MemoryReranker()
    items = [
        _make_item("old", "run_config", "src/old.py", 1.0),
        _make_item("new", "run_config", "src/new.py", 10.0),
        MemoryItem(
            id="layered",
            kind="run_config",
            pointer={},
            snippet="",
            dimensions={"layer": "L2"},
            stats={"created_at": 5.0},
        ),
    ]

    hints_recent = RerankHints(boost_dimensions=None, diversity_over=None, prefer_recent=True)
    reordered = reranker.rerank(items, hints_recent)
    assert reordered[0].id == "new"

    hints_boost = RerankHints(boost_dimensions={"layer": 1.0}, diversity_over=None, prefer_recent=False)
    reordered_boost = reranker.rerank(items, hints_boost)
    assert reordered_boost[0].id == "layered"


def test_selector_respects_recency_window() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "memory.db")
        store = MemoryStore(db_path=store_path)
        selector = MemorySelector(store=store, vector_store=None)

        old_item = _make_item("old", "run_config", "src/old.py", created_at=1.0)
        new_item = _make_item("new", "run_config", "src/new.py", created_at=10.0)
        store.upsert_item(old_item)
        store.upsert_item(new_item)

        profile = SelectorProfile(weights={}, per_kind_limit={}, recency_window=5)
        selected = selector.select(profile=profile, filters={})

        assert len(selected) == 1
        assert selected[0].id == "new"


def test_reranker_prefer_recent_toggle_changes_order() -> None:
    reranker = MemoryReranker()
    items = [
        _make_item("old", "run_config", "src/old.py", 1.0),
        _make_item("new", "run_config", "src/new.py", 10.0),
    ]

    reordered_recent = reranker.rerank(items, RerankHints(boost_dimensions=None, diversity_over=None, prefer_recent=True))
    assert reordered_recent[0].id == "new"

    reordered_no_recent = reranker.rerank(items, RerankHints(boost_dimensions=None, diversity_over=None, prefer_recent=False))
    assert reordered_no_recent[0].id == "old"
