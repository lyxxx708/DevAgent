from schemas.memory import MemoryItem, MemoryStats


def test_memory_items_and_stats():
    item = MemoryItem(
        id="m1",
        kind="error_pattern",
        pointer={"file": "a.py"},
        snippet="traceback...",
        dimensions={"layer": "L2"},
        stats={"score": 0.9},
    )
    stats = MemoryStats(counts_by_kind={"error_pattern": 1}, recent_activity_score=0.5)

    assert item.kind == "error_pattern"
    assert stats.counts_by_kind["error_pattern"] == 1
