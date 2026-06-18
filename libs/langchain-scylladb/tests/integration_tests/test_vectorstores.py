"""Standard LangChain integration tests for ScyllaDBVectorStore.

Runs the full VectorStoreIntegrationTests suite from langchain-tests against a
local ScyllaDB + vector-store stack managed by testcontainers (see conftest.py).
"""
from __future__ import annotations

from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_scylladb import ScyllaDBVectorStore


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
        try:
            yield store
        finally:
            store.delete_collection()
            store.close()

