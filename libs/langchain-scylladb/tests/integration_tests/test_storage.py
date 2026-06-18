"""Integration tests for ScyllaDB stores (local testcontainers)."""
from __future__ import annotations

import pytest
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy
from langchain_core.documents import Document

from langchain_scylladb.storage import ScyllaDBByteStore, ScyllaDBDocStore, ScyllaDBStore


@pytest.fixture()
def session(scylladb_service):
    info = scylladb_service
    profile = ExecutionProfile(
        load_balancing_policy=DCAwareRoundRobinPolicy(info["local_dc"])
    )
    cluster = Cluster(
        [info["host"]],
        port=info["port"],
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
        protocol_version=4,
    )
    s = cluster.connect()
    yield s, info["keyspace"]
    for tbl in ("store_test", "byte_store_test", "doc_store_test"):
        s.execute(f"DROP TABLE IF EXISTS {info['keyspace']}.{tbl}")
    cluster.shutdown()


# ---------------------------------------------------------------------------
# ScyllaDBStore
# ---------------------------------------------------------------------------


def test_store_mset_and_mget_roundtrip(session) -> None:
    db_session, keyspace = session
    store = ScyllaDBStore(db_session, keyspace=keyspace, table_name="store_test")

    store.mset([("k1", {"a": 1}), ("k2", [1, 2, 3])])
    result = store.mget(["k1", "k2", "missing"])

    assert result[0] == {"a": 1}
    assert result[1] == [1, 2, 3]
    assert result[2] is None


def test_store_mdelete_removes_only_specified_keys(session) -> None:
    db_session, keyspace = session
    store = ScyllaDBStore(db_session, keyspace=keyspace, table_name="store_test")

    store.mset([("del1", "v1"), ("keep1", "v2")])
    store.mdelete(["del1"])

    result = store.mget(["del1", "keep1"])
    assert result[0] is None
    assert result[1] == "v2"


def test_store_yield_keys(session) -> None:
    db_session, keyspace = session
    store = ScyllaDBStore(db_session, keyspace=keyspace, table_name="store_test")

    store.mset([("prefix_a", 1), ("prefix_b", 2), ("other", 3)])
    all_keys = list(store.yield_keys())
    prefix_keys = list(store.yield_keys(prefix="prefix"))

    assert "prefix_a" in all_keys
    assert "other" in all_keys
    assert set(prefix_keys) == {"prefix_a", "prefix_b"}


# ---------------------------------------------------------------------------
# ScyllaDBByteStore
# ---------------------------------------------------------------------------


def test_bytestore_roundtrip(session) -> None:
    db_session, keyspace = session
    store = ScyllaDBByteStore(db_session, keyspace=keyspace, table_name="byte_store_test")

    data = b"\x00\xff\xab\xcd"
    store.mset([("blob", data)])
    result = store.mget(["blob"])

    assert result[0] == data


# ---------------------------------------------------------------------------
# ScyllaDBDocStore
# ---------------------------------------------------------------------------


def test_docstore_roundtrip(session) -> None:
    db_session, keyspace = session
    store = ScyllaDBDocStore(db_session, keyspace=keyspace, table_name="doc_store_test")

    doc = Document(page_content="Hello ScyllaDB", metadata={"source": "test"})
    store.mset([("doc1", doc)])
    result = store.mget(["doc1"])

    assert result[0] is not None
    assert result[0].page_content == "Hello ScyllaDB"
    assert result[0].metadata["source"] == "test"


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_async_roundtrip(session) -> None:
    db_session, keyspace = session
    store = ScyllaDBStore(db_session, keyspace=keyspace, table_name="store_test")

    await store.amset([("async_k", "async_v")])
    result = await store.amget(["async_k"])

    assert result[0] == "async_v"


@pytest.mark.asyncio
async def test_store_async_delete(session) -> None:
    db_session, keyspace = session
    store = ScyllaDBStore(db_session, keyspace=keyspace, table_name="store_test")

    await store.amset([("to_del", "x")])
    await store.amdelete(["to_del"])
    result = await store.amget(["to_del"])

    assert result[0] is None


@pytest.mark.asyncio
async def test_store_ayield_keys(session) -> None:
    db_session, keyspace = session
    store = ScyllaDBStore(db_session, keyspace=keyspace, table_name="store_test")

    store.mset([("ay_1", 1), ("ay_2", 2)])
    keys = [k async for k in store.ayield_keys(prefix="ay_")]

    assert set(keys) == {"ay_1", "ay_2"}
