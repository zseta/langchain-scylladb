"""Unit tests for ScyllaDB index utilities."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from langchain_scylladb.index import (
    create_secondary_index,
    create_vector_index,
    drop_vector_index,
    list_indexes,
    wait_for_index,
)


# ---------------------------------------------------------------------------
# create_vector_index
# ---------------------------------------------------------------------------


def test_create_vector_index_executes_ddl() -> None:
    session = MagicMock()

    create_vector_index(session, "ks", "t", "embedding")

    session.execute.assert_called_once()
    ddl: str = session.execute.call_args[0][0]
    assert "vector_index" in ddl
    assert "embedding" in ddl
    assert "COSINE" in ddl


def test_create_vector_index_uses_custom_similarity() -> None:
    session = MagicMock()

    create_vector_index(session, "ks", "t", "vec", similarity="EUCLIDEAN")

    ddl: str = session.execute.call_args[0][0]
    assert "EUCLIDEAN" in ddl


def test_create_vector_index_default_name() -> None:
    session = MagicMock()
    create_vector_index(session, "ks", "t", "embedding")
    ddl: str = session.execute.call_args[0][0]
    assert "t_embedding_idx" in ddl


def test_create_vector_index_custom_name() -> None:
    session = MagicMock()
    create_vector_index(session, "ks", "t", "embedding", index_name="my_idx")
    ddl: str = session.execute.call_args[0][0]
    assert "my_idx" in ddl


def test_create_vector_index_extra_options() -> None:
    session = MagicMock()
    create_vector_index(session, "ks", "t", "vec", options={"M": "16"})
    ddl: str = session.execute.call_args[0][0]
    assert "M" in ddl


# ---------------------------------------------------------------------------
# drop_vector_index
# ---------------------------------------------------------------------------


def test_drop_vector_index_executes_drop_if_exists() -> None:
    session = MagicMock()

    drop_vector_index(session, "ks", "my_idx")

    session.execute.assert_called_once()
    ddl: str = session.execute.call_args[0][0]
    assert "DROP INDEX" in ddl
    assert "my_idx" in ddl


# ---------------------------------------------------------------------------
# create_secondary_index
# ---------------------------------------------------------------------------


def test_create_secondary_index_executes_ddl() -> None:
    session = MagicMock()
    create_secondary_index(session, "ks", "t", "col")
    ddl: str = session.execute.call_args[0][0]
    assert "CREATE INDEX" in ddl
    assert "col" in ddl


# ---------------------------------------------------------------------------
# wait_for_index
# ---------------------------------------------------------------------------


def test_wait_for_index_returns_immediately_when_present() -> None:
    session = MagicMock()
    row = MagicMock()
    row.index_name = "my_idx"
    session.execute.return_value = [row]

    # Should not raise
    wait_for_index(session, "ks", "t", "my_idx", timeout=5)


def test_wait_for_index_raises_timeout_when_index_absent() -> None:
    session = MagicMock()
    session.execute.return_value = []

    with pytest.raises(TimeoutError, match="my_idx"):
        wait_for_index(session, "ks", "t", "my_idx", timeout=0.01, poll_interval=0.005)


# ---------------------------------------------------------------------------
# list_indexes
# ---------------------------------------------------------------------------


def test_list_indexes_returns_index_names() -> None:
    session = MagicMock()
    row1 = MagicMock()
    row1.index_name = "idx_a"
    row2 = MagicMock()
    row2.index_name = "idx_b"
    session.execute.return_value = [row1, row2]

    result = list_indexes(session, "ks", "t")

    assert set(result) == {"idx_a", "idx_b"}


def test_list_indexes_empty() -> None:
    session = MagicMock()
    session.execute.return_value = []
    assert list_indexes(session, "ks", "t") == []
