"""Unit tests for ScyllaDBCache and ScyllaDBSemanticCache."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.outputs import Generation

from langchain_scylladb.cache import (
    ScyllaDBCache,
    _deserialize_generations,
    _serialize_generations,
)


def _make_cache(**kwargs) -> tuple[ScyllaDBCache, MagicMock]:
    fake_session = MagicMock()
    with patch.object(ScyllaDBCache, "_setup_schema"):
        cache = ScyllaDBCache(session=fake_session, **kwargs)
    return cache, fake_session


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def test_serialize_deserialize_roundtrip() -> None:
    gens = [Generation(text="hello")]
    result = _deserialize_generations(_serialize_generations(gens))
    assert result[0].text == "hello"


def test_serialize_multiple_generations() -> None:
    gens = [Generation(text="a"), Generation(text="b")]
    result = _deserialize_generations(_serialize_generations(gens))
    assert [g.text for g in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# ScyllaDBCache
# ---------------------------------------------------------------------------


def test_lookup_returns_none_on_miss() -> None:
    cache, fake_session = _make_cache()
    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.return_value.one.return_value = None

    assert cache.lookup("prompt", "llm") is None


def test_lookup_deserializes_on_hit() -> None:
    cache, fake_session = _make_cache()
    gens = [Generation(text="answer")]
    fake_row = MagicMock()
    fake_row.return_val = _serialize_generations(gens)
    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.return_value.one.return_value = fake_row

    result = cache.lookup("prompt", "llm")

    assert result is not None
    assert result[0].text == "answer"


def test_update_performs_parameterised_insert() -> None:
    cache, fake_session = _make_cache()
    fake_session.prepare.return_value = MagicMock()

    cache.update("p", "l", [Generation(text="result")])

    fake_session.prepare.assert_called_once()
    prepared_query: str = fake_session.prepare.call_args[0][0]
    assert "INSERT" in prepared_query
    assert "?" in prepared_query
    fake_session.execute.assert_called_once()


def test_clear_issues_truncate() -> None:
    cache, fake_session = _make_cache()

    cache.clear()

    call_arg = fake_session.execute.call_args[0][0]
    assert "TRUNCATE" in call_arg


def test_setup_schema_creates_table() -> None:
    fake_session = MagicMock()
    ScyllaDBCache(session=fake_session)
    ddl: str = fake_session.execute.call_args[0][0]
    assert "CREATE TABLE" in ddl
    assert "prompt" in ddl
    assert "llm_string" in ddl


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alookup_delegates_to_lookup() -> None:
    cache, fake_session = _make_cache()
    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.return_value.one.return_value = None

    assert await cache.alookup("p", "l") is None


@pytest.mark.asyncio
async def test_aupdate_delegates_to_update() -> None:
    cache, fake_session = _make_cache()
    fake_session.prepare.return_value = MagicMock()
    await cache.aupdate("p", "l", [Generation(text="x")])
    fake_session.execute.assert_called()


@pytest.mark.asyncio
async def test_aclear_delegates_to_clear() -> None:
    cache, fake_session = _make_cache()
    await cache.aclear()
    call_arg = fake_session.execute.call_args[0][0]
    assert "TRUNCATE" in call_arg
