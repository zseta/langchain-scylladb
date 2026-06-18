"""ScyllaDB LLM caches: exact-match and semantic (vector-similarity)."""
from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any, List, Optional

from langchain_core.caches import BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation

from cassandra.cluster import Session

logger = logging.getLogger(__name__)


def _serialize_generations(generations: List[Generation]) -> str:
    return json.dumps([dumps(g) for g in generations])


def _deserialize_generations(s: str) -> List[Generation]:
    return [loads(g) for g in json.loads(s)]


class ScyllaDBCache(BaseCache):
    """Exact-match LLM response cache backed by ScyllaDB.

    CQL schema::

        CREATE TABLE {keyspace}.{table} (
            prompt      TEXT,
            llm_string  TEXT,
            return_val  TEXT,
            PRIMARY KEY ((prompt), llm_string)
        );

    Args:
        session: An open ``cassandra.cluster.Session``.
        keyspace: ScyllaDB keyspace (must already exist).
        table_name: Table name within the keyspace.
    """

    def __init__(
        self,
        session: Session,
        *,
        keyspace: str = "langchain_scylladb",
        table_name: str = "llm_cache",
    ) -> None:
        self._session = session
        self._keyspace = keyspace
        self._table_name = table_name
        self._setup_schema()

    def _setup_schema(self) -> None:
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._table_name} (
                prompt      TEXT,
                llm_string  TEXT,
                return_val  TEXT,
                PRIMARY KEY ((prompt), llm_string)
            )
            """
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
        stmt = self._session.prepare(
            f"SELECT return_val FROM {self._keyspace}.{self._table_name} "
            f"WHERE prompt = ? AND llm_string = ?"
        )
        row = self._session.execute(stmt, (prompt, llm_string)).one()
        if row is None:
            return None
        return _deserialize_generations(row.return_val)

    async def alookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
        return self.lookup(prompt, llm_string)

    def update(self, prompt: str, llm_string: str, return_val: List[Generation]) -> None:
        stmt = self._session.prepare(
            f"INSERT INTO {self._keyspace}.{self._table_name} "
            f"(prompt, llm_string, return_val) VALUES (?, ?, ?)"
        )
        self._session.execute(stmt, (prompt, llm_string, _serialize_generations(return_val)))

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: List[Generation]
    ) -> None:
        self.update(prompt, llm_string, return_val)

    def clear(self, **kwargs: Any) -> None:
        self._session.execute(f"TRUNCATE {self._keyspace}.{self._table_name}")

    async def aclear(self, **kwargs: Any) -> None:
        self.clear(**kwargs)


class ScyllaDBSemanticCache(BaseCache):
    """Semantic LLM response cache backed by ScyllaDB vector search.

    Requires Vector Search enabled ScyllaDB cluster. On a hit the
    most-similar prompt that was previously cached is returned, provided its
    similarity score meets *score_threshold*.

    ``llm_string`` is stored as an MD5 hash in the vector-store metadata so
    that very long LLM serialisations do not inflate row sizes.  A per-instance
    ``lru_cache`` avoids double-embedding the same prompt during a
    ``lookup`` → ``update`` cycle.

    Args:
        session: An open ``cassandra.cluster.Session``.
        embedding: LangChain ``Embeddings`` instance.
        keyspace: ScyllaDB keyspace (must already exist).
        table_name: Vector-store table name.
        score_threshold: Minimum similarity score to consider a cache hit
            (0–1 for COSINE, higher is more similar).
        **vectorstore_kwargs: Extra keyword arguments forwarded to
            ``ScyllaDBVectorStore``.
    """

    def __init__(
        self,
        session: Session,
        embedding: Embeddings,
        *,
        keyspace: str = "langchain_scylladb",
        table_name: str = "semantic_cache",
        score_threshold: float = 0.9,
        **vectorstore_kwargs: Any,
    ) -> None:
        from langchain_scylladb.vectorstores import ScyllaDBVectorStore

        self._score_threshold = score_threshold
        self._vectorstore = ScyllaDBVectorStore(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            **vectorstore_kwargs,
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
        llm_hash = hashlib.md5(llm_string.encode()).hexdigest()  # noqa: S324
        results = self._vectorstore.similarity_search_with_score(
            prompt,
            k=1,
            filter={"llm_string_hash": llm_hash},
        )
        if not results:
            return None
        doc, score = results[0]
        if score < self._score_threshold:
            return None
        return _deserialize_generations(doc.metadata["return_val"])

    async def alookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
        return self.lookup(prompt, llm_string)

    def update(self, prompt: str, llm_string: str, return_val: List[Generation]) -> None:
        llm_hash = hashlib.md5(llm_string.encode()).hexdigest()  # noqa: S324
        metadata = {
            "llm_string_hash": llm_hash,
            "return_val": _serialize_generations(return_val),
        }
        self._vectorstore.add_texts([prompt], metadatas=[metadata])

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: List[Generation]
    ) -> None:
        self.update(prompt, llm_string, return_val)

    def clear(self, **kwargs: Any) -> None:
        self._vectorstore.delete_collection()

    async def aclear(self, **kwargs: Any) -> None:
        self.clear(**kwargs)
