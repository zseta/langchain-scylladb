"""Standard LangChain integration tests for ScyllaDBVectorStore.

Runs the full VectorStoreIntegrationTests suite from langchain-tests against a
local ScyllaDB + vector-store stack managed by testcontainers (see conftest.py).
"""
from __future__ import annotations

import time
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_scylladb import ScyllaDBVectorStore


def _delay_ann_reads(store: ScyllaDBVectorStore, delay: float = 2.0) -> None:
    similarity_search_with_score_by_vector = store.similarity_search_with_score_by_vector
    similarity_search_with_embeddings = store._similarity_search_with_embeddings

    def delayed_similarity_search_with_score_by_vector(*args, **kwargs):
        time.sleep(delay)
        return similarity_search_with_score_by_vector(*args, **kwargs)

    def delayed_similarity_search_with_embeddings(*args, **kwargs):
        time.sleep(delay)
        return similarity_search_with_embeddings(*args, **kwargs)

    store.similarity_search_with_score_by_vector = delayed_similarity_search_with_score_by_vector
    store._similarity_search_with_embeddings = delayed_similarity_search_with_embeddings


class TestScyllaDBVectorStore(VectorStoreIntegrationTests):
    """Runs the full standard LangChain VectorStore test suite."""

    @pytest.fixture()
    def vectorstore(self, scylladb_service: dict) -> Generator[VectorStore, None, None]:
        store = ScyllaDBVectorStore(
            embedding=self.get_embeddings(),
            host=scylladb_service["host"],
            port=scylladb_service["port"],
            local_dc=scylladb_service["local_dc"],
            keyspace=scylladb_service["keyspace"],
            table_name="langchain_test",
        )
        _delay_ann_reads(store)
        try:
            yield store
        finally:
            store.delete_collection()
            store.close()

