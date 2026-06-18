"""Unit tests for ScyllaDBVectorStore and helpers.

These tests run without a real ScyllaDB cluster — the driver session and schema
setup are replaced by mocks.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings

from langchain_scylladb import ScyllaDBVectorStore
from langchain_scylladb.vectorstores import _matches_filter


# ---------------------------------------------------------------------------
# Minimal fake embeddings (no I/O)
# ---------------------------------------------------------------------------


class FakeEmbeddings(Embeddings):
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(i)] * self.dim for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * self.dim

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


# ---------------------------------------------------------------------------
# Helpers to build a ScyllaDBVectorStore without a real DB
# ---------------------------------------------------------------------------


def _make_store(**kwargs) -> ScyllaDBVectorStore:
    """Return a ScyllaDBVectorStore with all I/O mocked out."""
    fake_session = MagicMock()
    fake_cluster = MagicMock()

    with patch(
        "langchain_scylladb.vectorstores._make_session",
        return_value=(fake_cluster, fake_session),
    ), patch.object(ScyllaDBVectorStore, "_setup_schema"):
        store = ScyllaDBVectorStore(
            embedding=FakeEmbeddings(),
            **kwargs,
        )
    return store


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_similarity_function_raises() -> None:
    with pytest.raises(ValueError, match="similarity_function"):
        _make_store(similarity_function="INVALID")


@pytest.mark.parametrize("fn", ["COSINE", "DOT_PRODUCT", "EUCLIDEAN"])
def test_valid_similarity_functions(fn: str) -> None:
    store = _make_store(similarity_function=fn)
    assert store._similarity_function == fn


def test_default_attributes() -> None:
    store = _make_store()
    assert store._keyspace == "langchain_scylladb"
    assert store._table_name == "documents"
    assert store._similarity_function == "COSINE"
    assert store._index_delay == 1.5
    assert store._embedding_dimension == 8  # FakeEmbeddings.dim


def test_custom_attributes() -> None:
    store = _make_store(
        keyspace="my_ks",
        table_name="my_tbl",
        similarity_function="EUCLIDEAN",
        index_delay=0.0,
    )
    assert store._keyspace == "my_ks"
    assert store._table_name == "my_tbl"
    assert store._similarity_function == "EUCLIDEAN"
    assert store._index_delay == 0.0


def test_owns_cluster_when_no_session_provided() -> None:
    store = _make_store()
    assert store._owns_cluster is True


def test_does_not_own_cluster_when_session_provided() -> None:
    fake_session = MagicMock()
    with patch.object(ScyllaDBVectorStore, "_setup_schema"):
        store = ScyllaDBVectorStore(
            embedding=FakeEmbeddings(),
            session=fake_session,
        )
    assert store._owns_cluster is False


# ---------------------------------------------------------------------------
# _matches_filter
# ---------------------------------------------------------------------------


class TestMatchesFilter:
    def test_equality_match(self) -> None:
        assert _matches_filter({"source": "docs"}, {"source": "docs"})

    def test_equality_no_match(self) -> None:
        assert not _matches_filter({"source": "docs"}, {"source": "blog"})

    def test_missing_key_no_match(self) -> None:
        assert not _matches_filter({}, {"source": "docs"})

    def test_empty_filter_matches_everything(self) -> None:
        assert _matches_filter({"source": "docs"}, {})
        assert _matches_filter({}, {})

    def test_multiple_keys_all_match(self) -> None:
        meta = {"source": "docs", "lang": "en"}
        assert _matches_filter(meta, {"source": "docs", "lang": "en"})

    def test_multiple_keys_partial_miss(self) -> None:
        meta = {"source": "docs", "lang": "en"}
        assert not _matches_filter(meta, {"source": "docs", "lang": "fr"})

    # $and
    def test_and_all_match(self) -> None:
        meta = {"a": 1, "b": 2}
        assert _matches_filter(meta, {"$and": [{"a": 1}, {"b": 2}]})

    def test_and_partial_miss(self) -> None:
        meta = {"a": 1, "b": 2}
        assert not _matches_filter(meta, {"$and": [{"a": 1}, {"b": 99}]})

    def test_and_empty_list(self) -> None:
        # all() of empty is True
        assert _matches_filter({"a": 1}, {"$and": []})

    # $or
    def test_or_one_matches(self) -> None:
        meta = {"source": "blog"}
        assert _matches_filter(meta, {"$or": [{"source": "docs"}, {"source": "blog"}]})

    def test_or_none_match(self) -> None:
        meta = {"source": "other"}
        assert not _matches_filter(meta, {"$or": [{"source": "docs"}, {"source": "blog"}]})

    def test_or_empty_list(self) -> None:
        # any() of empty is False
        assert not _matches_filter({"a": 1}, {"$or": []})

    # $not
    def test_not_negates_match(self) -> None:
        assert _matches_filter({"source": "blog"}, {"$not": {"source": "docs"}})

    def test_not_negates_miss(self) -> None:
        assert not _matches_filter({"source": "docs"}, {"$not": {"source": "docs"}})

    # nested
    def test_nested_and_or(self) -> None:
        meta = {"type": "article", "lang": "en"}
        f = {"$and": [{"type": "article"}, {"$or": [{"lang": "en"}, {"lang": "fr"}]}]}
        assert _matches_filter(meta, f)

    def test_nested_and_or_no_match(self) -> None:
        meta = {"type": "article", "lang": "de"}
        f = {"$and": [{"type": "article"}, {"$or": [{"lang": "en"}, {"lang": "fr"}]}]}
        assert not _matches_filter(meta, f)
