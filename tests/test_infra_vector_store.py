from __future__ import annotations

from infra.vector_store import VectorStore


def test_vector_store_add_search_delete_python_path() -> None:
    store = VectorStore(dim=3, use_faiss=False)
    ids = ["a", "b", "c"]
    vectors = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
    store.add(ids, vectors)

    results = store.search([0.1, 0.1, 0.1], k=2)
    assert results[0][0] == "a"

    store.delete(["a"])
    assert store.count() == 2
    results_after_delete = store.search([0.1, 0.1, 0.1], k=3)
    returned_ids = {res[0] for res in results_after_delete}
    assert "a" not in returned_ids
