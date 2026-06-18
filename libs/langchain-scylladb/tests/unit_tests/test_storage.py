"""Unit tests for ScyllaDBStore, ScyllaDBByteStore."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_scylladb.storage import ScyllaDBByteStore, ScyllaDBStore


def _make_store(cls, **kwargs):
    fake_session = MagicMock()
    with patch.object(cls, "_setup_schema"):
        store = cls(session=fake_session, **kwargs)
    return store, fake_session


# ---------------------------------------------------------------------------
# ScyllaDBStore — mget
# ---------------------------------------------------------------------------


def test_mget_returns_none_for_missing_keys() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.return_value.one.return_value = None

    result = store.mget(["missing"])

    assert result == [None]


def test_mget_reassembles_results_in_key_order() -> None:
    store, fake_session = _make_store(ScyllaDBStore)

    def mock_execute(stmt, params):
        key = params[0]
        mapping = {"a": "val_a", "c": "val_c"}
        if key in mapping:
            row = MagicMock()
            row.value = json.dumps(mapping[key])
            return MagicMock(one=lambda: row)
        return MagicMock(one=lambda: None)

    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.side_effect = mock_execute

    result = store.mget(["a", "b", "c"])

    assert result == ["val_a", None, "val_c"]


def test_mget_empty_keys_returns_empty_list() -> None:
    store, _ = _make_store(ScyllaDBStore)
    assert store.mget([]) == []


# ---------------------------------------------------------------------------
# ScyllaDBStore — mset
# ---------------------------------------------------------------------------


def test_mset_issues_batch_inserts() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    fake_session.prepare.return_value = MagicMock()

    store.mset([("k1", "v1"), ("k2", "v2")])

    fake_session.prepare.assert_called_once()
    insert_stmt: str = fake_session.prepare.call_args[0][0]
    assert "INSERT" in insert_stmt
    assert "?" in insert_stmt
    # One execute call for the single batch
    fake_session.execute.assert_called_once()


def test_mset_empty_does_nothing() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    store.mset([])
    fake_session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# ScyllaDBStore — mdelete
# ---------------------------------------------------------------------------


def test_mdelete_issues_delete_per_key() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    fake_session.prepare.return_value = MagicMock()

    store.mdelete(["k1", "k2"])

    assert fake_session.execute.call_count == 2


def test_mdelete_empty_does_nothing() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    store.mdelete([])
    fake_session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# ScyllaDBStore — yield_keys
# ---------------------------------------------------------------------------


def test_yield_keys_iterates_all_keys() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    fake_session.execute.return_value = [MagicMock(key="k1"), MagicMock(key="k2")]

    keys = list(store.yield_keys())

    assert set(keys) == {"k1", "k2"}


def test_yield_keys_with_prefix_filters_correctly() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    fake_session.execute.return_value = [
        MagicMock(key="foo_1"),
        MagicMock(key="bar_2"),
        MagicMock(key="foo_3"),
    ]

    keys = list(store.yield_keys(prefix="foo"))

    assert set(keys) == {"foo_1", "foo_3"}


# ---------------------------------------------------------------------------
# ScyllaDBByteStore — encode/decode roundtrip
# ---------------------------------------------------------------------------


def test_bytestore_roundtrip() -> None:
    store, _ = _make_store(ScyllaDBByteStore)
    data = b"\x00\xff\xab\xcd"
    assert store._decode_value(store._encode_value(data)) == data


def test_bytestore_encode_is_ascii() -> None:
    store, _ = _make_store(ScyllaDBByteStore)
    encoded = store._encode_value(b"hello")
    assert isinstance(encoded, str)
    encoded.encode("ascii")  # should not raise


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_amget_delegates_to_mget() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.return_value.one.return_value = None

    result = await store.amget(["k"])

    assert result == [None]


@pytest.mark.asyncio
async def test_amset_delegates_to_mset() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    fake_session.prepare.return_value = MagicMock()
    await store.amset([("k", "v")])
    fake_session.execute.assert_called()


@pytest.mark.asyncio
async def test_amdelete_delegates_to_mdelete() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    fake_session.prepare.return_value = MagicMock()
    await store.amdelete(["k"])
    fake_session.execute.assert_called()


@pytest.mark.asyncio
async def test_ayield_keys_yields_all() -> None:
    store, fake_session = _make_store(ScyllaDBStore)
    fake_session.execute.return_value = [MagicMock(key="a"), MagicMock(key="b")]
    keys = [k async for k in store.ayield_keys()]
    assert set(keys) == {"a", "b"}
