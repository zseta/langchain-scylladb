"""Integration tests for ScyllaDBCache (local) and ScyllaDBSemanticCache (cloud)."""
from __future__ import annotations

import pytest
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy
from langchain_core.outputs import Generation

from langchain_scylladb.cache import ScyllaDBCache

# Cloud semantic cache tests are marked with @pytest.mark.cloud


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
    s.execute(f"DROP TABLE IF EXISTS {info['keyspace']}.cache_test")
    cluster.shutdown()


def _make_cache(db_session, keyspace):
    return ScyllaDBCache(db_session, keyspace=keyspace, table_name="cache_test")


# ---------------------------------------------------------------------------
# Exact-match cache
# ---------------------------------------------------------------------------


def test_exact_cache_write_then_read(session) -> None:
    db_session, keyspace = session
    cache = _make_cache(db_session, keyspace)

    gens = [Generation(text="answer")]
    cache.update("prompt1", "llm1", gens)
    result = cache.lookup("prompt1", "llm1")

    assert result is not None
    assert result[0].text == "answer"


def test_exact_cache_different_llm_string_is_miss(session) -> None:
    db_session, keyspace = session
    cache = _make_cache(db_session, keyspace)

    cache.update("prompt2", "llm_a", [Generation(text="cached")])
    result = cache.lookup("prompt2", "llm_b")

    assert result is None


def test_exact_cache_clear_causes_miss(session) -> None:
    db_session, keyspace = session
    cache = _make_cache(db_session, keyspace)

    cache.update("prompt3", "llm1", [Generation(text="to clear")])
    cache.clear()
    result = cache.lookup("prompt3", "llm1")

    assert result is None


# ---------------------------------------------------------------------------
# Semantic cache — cloud only
# ---------------------------------------------------------------------------


@pytest.mark.cloud
def test_semantic_cache_similar_prompt_returns_cached(cloud_session) -> None:
    from typing import List

    from langchain_core.embeddings import Embeddings

    from langchain_scylladb.cache import ScyllaDBSemanticCache

    class _FakeEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

        def embed_query(self, text: str) -> List[float]:
            return [0.5, 0.5, 0.5, 0.5]

    cache = ScyllaDBSemanticCache(
        cloud_session,
        _FakeEmbeddings(),
        keyspace="langchain_test",
        table_name="semantic_cache_test",
        score_threshold=0.5,
        embedding_dimension=4,
    )
    try:
        import time
        cache.update("hello world", "llm1", [Generation(text="cached result")])
        time.sleep(2)  # wait for HNSW index
        result = cache.lookup("hello world", "llm1")
        assert result is not None
        assert result[0].text == "cached result"
    finally:
        cache.clear()
