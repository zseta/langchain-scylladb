"""Integration tests for ScyllaDB index utilities (cloud only for vector index)."""
from __future__ import annotations

import pytest
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy

from langchain_scylladb.index import (
    create_secondary_index,
    create_vector_index,
    drop_vector_index,
    list_indexes,
    wait_for_index,
)

_TABLE = "index_test"


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
    ks = info["keyspace"]
    s.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ks}.{_TABLE} (
            id   TEXT PRIMARY KEY,
            tag  TEXT
        )
        """
    )
    yield s, ks
    s.execute(f"DROP TABLE IF EXISTS {ks}.{_TABLE}")
    cluster.shutdown()


def test_create_and_list_secondary_index(session) -> None:
    db_session, keyspace = session
    create_secondary_index(db_session, keyspace, _TABLE, "tag", index_name="tag_idx")

    indexes = list_indexes(db_session, keyspace, _TABLE)
    assert "tag_idx" in indexes


def test_wait_for_secondary_index(session) -> None:
    db_session, keyspace = session
    create_secondary_index(db_session, keyspace, _TABLE, "tag", index_name="wait_idx")
    # Should not raise — index was just created
    wait_for_index(db_session, keyspace, _TABLE, "wait_idx", timeout=30)


# ---------------------------------------------------------------------------
# Vector index — cloud only (HNSW requires vector-store service)
# ---------------------------------------------------------------------------


@pytest.mark.cloud
def test_create_vector_index_roundtrip(cloud_session) -> None:
    ks = "langchain_test"
    tbl = "vector_idx_test"
    try:
        cloud_session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ks}.{tbl} (
                id        TEXT PRIMARY KEY,
                embedding vector<float, 4>
            )
            """
        )
        create_vector_index(cloud_session, ks, tbl, "embedding", index_name="vec_idx_test")
        wait_for_index(cloud_session, ks, tbl, "vec_idx_test", timeout=60)

        indexes = list_indexes(cloud_session, ks, tbl)
        assert "vec_idx_test" in indexes

        drop_vector_index(cloud_session, ks, "vec_idx_test")
        indexes_after = list_indexes(cloud_session, ks, tbl)
        assert "vec_idx_test" not in indexes_after
    finally:
        cloud_session.execute(f"DROP TABLE IF EXISTS {ks}.{tbl}")
