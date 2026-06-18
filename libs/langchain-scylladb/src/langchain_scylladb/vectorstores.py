from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Iterable, Optional

import numpy as np
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from cassandra.cluster import Cluster, Session, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy
from cassandra.query import SimpleStatement
from cassandra import InvalidRequest


_SIMILARITY_FUNCTIONS = {"COSINE", "DOT_PRODUCT", "EUCLIDEAN"}

_SIMILARITY_SCORE_FN = {
    "COSINE": "similarity_cosine",
    "DOT_PRODUCT": "similarity_dot_product",
    "EUCLIDEAN": "similarity_euclidean",
}


def _make_session(
    host: str,
    port: int,
    username: Optional[str],
    password: Optional[str],
    local_dc: Optional[str],
) -> tuple[Cluster, Session]:
    auth = (
        PlainTextAuthProvider(username=username, password=password)
        if username and password
        else None
    )
    policy = DCAwareRoundRobinPolicy(local_dc=local_dc) if local_dc else DCAwareRoundRobinPolicy()
    profile = ExecutionProfile(load_balancing_policy=policy)
    cluster = Cluster(
        contact_points=[host],
        port=port,
        auth_provider=auth,
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
    )
    return cluster, cluster.connect()


class ScyllaDBVectorStore(VectorStore):
    """LangChain VectorStore backed by ScyllaDB's native HNSW vector index.

    Stores documents with embeddings in a ScyllaDB table and queries
    them using Approximate Nearest Neighbour (ANN) search.

    Args:
        embedding: LangChain Embeddings instance used to embed texts.
        session: An existing ``cassandra.cluster.Session`` connected to
            ScyllaDB.  When *None*, a new connection is opened using the
            *host* / *port* / *username* / *password* / *local_dc*
            parameters.
        keyspace: ScyllaDB keyspace.  Created automatically if absent.
        table_name: Table name within the keyspace.
        host: ScyllaDB contact point (used when *session* is *None*).
        port: CQL port (default 9042).
        username: CQL username.
        password: CQL password.
        local_dc: data center name (e.g. `AWS_US_EAST_1`).
        embedding_dimension: Dimensionality of the embedding vectors.
            Detected automatically from the embedding model if *None*.
        similarity_function: One of ``COSINE`` (default), ``DOT_PRODUCT``,
            or ``EUCLIDEAN``.
        index_delay: Seconds to wait after writes to allow the HNSW index to
            catch up via CDC (default 1.5).  Set to 0 to disable.
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        session: Optional[Session] = None,
        keyspace: str = "langchain_scylladb",
        table_name: str = "documents",
        host: str = "localhost",
        port: int = 9042,
        username: Optional[str] = None,
        password: Optional[str] = None,
        local_dc: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        similarity_function: str = "COSINE",
        index_delay: float = 1.5,
    ) -> None:
        if similarity_function not in _SIMILARITY_FUNCTIONS:
            raise ValueError(
                f"similarity_function must be one of {_SIMILARITY_FUNCTIONS}, "
                f"got {similarity_function!r}"
            )

        self._embedding = embedding
        self._keyspace = keyspace
        self._table_name = table_name
        self._similarity_function = similarity_function
        self._score_fn = _SIMILARITY_SCORE_FN[similarity_function]
        self._index_delay = index_delay

        # Detect embedding dimension lazily if not provided
        if embedding_dimension is None:
            sample = embedding.embed_query("probe")
            embedding_dimension = len(sample)
        self._embedding_dimension = embedding_dimension

        # Connection management
        self._owns_cluster = session is None
        if session is None:
            self._cluster, self._session = _make_session(
                host, port, username, password, local_dc
            )
        else:
            self._cluster = None  # type: ignore[assignment]
            self._session = session

        self._setup_schema()
        # Give the vector-store sidecar time to register the new HNSW index.
        if self._index_delay > 0:
            time.sleep(self._index_delay)

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _setup_schema(self) -> None:
        """Create keyspace, table, and vector index if they don't exist."""
        self._session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {self._keyspace}
            WITH replication = {{'class': 'NetworkTopologyStrategy', 'replication_factor': 3}}
            AND tablets = {{'enabled': true}}
            """
        )
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._table_name} (
                id        text PRIMARY KEY,
                content   text,
                metadata  text,
                embedding vector<float, {self._embedding_dimension}>
            )
            """
        )
        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._table_name}_embedding_idx
            ON {self._keyspace}.{self._table_name}(embedding)
            USING 'vector_index'
            WITH OPTIONS = {{'similarity_function': '{self._similarity_function}'}}
            """
        )

    def delete_collection(self) -> None:
        """Permanently drop the table and its vector index.

        The store is unusable after this call.
        """
        self._session.execute(
            f"DROP INDEX IF EXISTS {self._keyspace}.{self._table_name}_embedding_idx"
        )
        self._session.execute(
            f"DROP TABLE IF EXISTS {self._keyspace}.{self._table_name}"
        )

    def close(self) -> None:
        """Shut down the underlying Cluster connection if we own it."""
        if self._owns_cluster and self._cluster is not None:
            self._cluster.shutdown()

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """ScyllaDB's similarity functions (``similarity_cosine``,
        ``similarity_dot_product``, ``similarity_euclidean``) all return
        scores in [0, 1] where 1 = most similar.
        """
        return lambda score: score

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        texts_list = list(texts)
        if not texts_list:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        else:
            # Some callers pass a list that contains None for documents without
            # an explicit ID (e.g. langchain_core's add_documents when mixing
            # Document objects with and without .id set).  Generate UUIDs for
            # those slots.
            ids = [i if i is not None else str(uuid.uuid4()) for i in ids]
        metadatas = metadatas or [{} for _ in texts_list]
        vectors = self._embedding.embed_documents(texts_list)

        insert_stmt = self._session.prepare(
            f"""
            INSERT INTO {self._keyspace}.{self._table_name}
                (id, content, metadata, embedding)
            VALUES (?, ?, ?, ?)
            """
        )
        for doc_id, text, meta, vector in zip(ids, texts_list, metadatas, vectors):
            self._session.execute(
                insert_stmt,
                (doc_id, text, json.dumps(meta), list(vector)),
            )

        # ScyllaDB's HNSW vector index propagates through CDC asynchronously.
        # Wait briefly so that immediately-following ANN queries see the new rows.
        if self._index_delay > 0:
            time.sleep(self._index_delay)

        return ids

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from the store by their IDs.

        When *ids* is ``None`` or empty, all documents are removed.
        ``TRUNCATE`` is avoided because it does not propagate to the
        CDC-based HNSW index; the table is dropped and recreated instead.
        When *ids* is provided, only those specific documents are deleted.

        Returns ``True`` on success.
        """
        if not ids:
            self._session.execute(
                f"DROP INDEX IF EXISTS {self._keyspace}.{self._table_name}_embedding_idx"
            )
            self._session.execute(
                f"DROP TABLE IF EXISTS {self._keyspace}.{self._table_name}"
            )
            self._setup_schema()
            if self._index_delay > 0:
                time.sleep(self._index_delay)
            return True
        delete_stmt = self._session.prepare(
            f"DELETE FROM {self._keyspace}.{self._table_name} WHERE id = ?"
        )
        for doc_id in ids:
            self._session.execute(delete_stmt, (doc_id,))

        if self._index_delay > 0:
            time.sleep(self._index_delay)

        return True

    def get_by_ids(self, ids: Iterable[str]) -> list[Document]:
        ids_list = list(ids)
        if not ids_list:
            return []
        select_stmt = self._session.prepare(
            f"SELECT id, content, metadata FROM {self._keyspace}.{self._table_name} WHERE id = ?"
        )
        docs: list[Document] = []
        for doc_id in ids_list:
            row = self._session.execute(select_stmt, (doc_id,)).one()
            if row is not None:
                docs.append(
                    Document(
                        id=str(row.id),
                        page_content=row.content,
                        metadata=json.loads(row.metadata),
                    )
                )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to an embedding vector, with scores.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of (Document, score) tuples; score is in [0, 1] (higher = more similar).
        """
        # Fetch extra candidates when filtering so we have enough after pruning
        fetch_k = k if filter is None else k * 10
        try:
            rows = self._session.execute(
                SimpleStatement(
                    f"""
                    SELECT id, content, metadata,
                           {self._score_fn}(embedding, %s) AS score
                    FROM {self._keyspace}.{self._table_name}
                    ORDER BY embedding ANN OF %s
                    LIMIT {fetch_k}
                    """
                ),
                (list(embedding), list(embedding)),
            )
        except InvalidRequest as _e:
            err = str(_e)
            if "missing index" in err:
                # The vector-store sidecar hasn't registered the newly-created
                # HNSW index yet.  Retry a few times before giving up.
                for _retry in range(10):
                    time.sleep(2.0)
                    try:
                        rows = self._session.execute(
                            SimpleStatement(
                                f"""
                    SELECT id, content, metadata,
                           {self._score_fn}(embedding, %s) AS score
                    FROM {self._keyspace}.{self._table_name}
                    ORDER BY embedding ANN OF %s
                    LIMIT {fetch_k}
                    """
                            ),
                            (list(embedding), list(embedding)),
                        )
                        break  # success
                    except InvalidRequest as _e2:
                        if "missing index" not in str(_e2) or _retry == 9:
                            return []
                else:
                    return []
            else:
                # ANN query on an empty table raises InvalidRequest.
                return []
        results: list[tuple[Document, float]] = []
        for row in rows:
            meta = json.loads(row.metadata)
            if filter is not None and not _matches_filter(meta, filter):
                continue
            doc = Document(
                id=str(row.id),
                page_content=row.content,
                metadata=meta,
            )
            results.append((doc, float(row.score)))
            if len(results) >= k:
                break
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        query_vector = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            query_vector, k=k, filter=filter, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to an embedding vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            filter: Optional metadata filter.
        """
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding, k=k, filter=filter, **kwargs
            )
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        return [
            doc
            for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        ]

    def _similarity_search_with_embeddings(
        self,
        query_vector: list[float],
        k: int = 4,
        filter: Optional[dict] = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float, list[float]]]:
        """Return (doc, score, embedding) tuples for the top-fetch_k candidates."""
        actual_fetch = fetch_k if filter is None else fetch_k * 10
        try:
            rows = self._session.execute(
                SimpleStatement(
                    f"""
                    SELECT id, content, metadata, embedding,
                           {self._score_fn}(embedding, %s) AS score
                    FROM {self._keyspace}.{self._table_name}
                    ORDER BY embedding ANN OF %s
                    LIMIT {actual_fetch}
                    """
                ),
                (list(query_vector), list(query_vector)),
            )
        except InvalidRequest:
            return []
        results: list[tuple[Document, float, list[float]]] = []
        for row in rows:
            meta = json.loads(row.metadata)
            if filter is not None and not _matches_filter(meta, filter):
                continue
            doc = Document(
                id=str(row.id),
                page_content=row.content,
                metadata=meta,
            )
            results.append((doc, float(row.score), list(row.embedding)))
            if len(results) >= fetch_k:
                break
        return results

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs using Maximal Marginal Relevance given an embedding vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            fetch_k: Number of ANN candidates to retrieve before re-ranking.
            lambda_mult: Trade-off between relevance and diversity (0 = max diversity, 1 = max relevance).
            filter: Optional metadata filter.
        """
        candidates = self._similarity_search_with_embeddings(
            embedding, k=k, filter=filter, fetch_k=fetch_k
        )
        if not candidates:
            return []
        docs = [doc for doc, _, _ in candidates]
        embeddings_arr = np.array([emb for _, _, emb in candidates], dtype=np.float32)
        query_arr = np.array(embedding, dtype=np.float32)
        selected_indices = maximal_marginal_relevance(
            query_arr, embeddings_arr, lambda_mult=lambda_mult, k=k
        )
        return [docs[i] for i in selected_indices]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs using Maximal Marginal Relevance.

        MMR balances relevance (similarity to the query) against diversity
        (dissimilarity from already-selected documents), controlled by
        *lambda_mult* (0 = max diversity, 1 = max relevance).

        Args:
            query: Text query.
            k: Number of results to return.
            fetch_k: Number of ANN candidates to retrieve before re-ranking.
            lambda_mult: Trade-off between relevance and diversity.
            filter: Optional metadata filter.
        """
        query_vector = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_vector, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )

    def replace_metadata(
        self,
        id_to_metadata: dict[str, dict],
    ) -> None:
        """Replace the metadata of documents identified by their IDs.

        The new metadata dict completely replaces the existing one for each ID.
        IDs that do not correspond to an existing document will result in a
        partial row (no content or embedding) that is unreachable via vector
        search.

        Args:
            id_to_metadata: Mapping of document ID → new metadata dict.
        """
        update_stmt = self._session.prepare(
            f"UPDATE {self._keyspace}.{self._table_name} SET metadata = ? WHERE id = ?"
        )
        for doc_id, meta in id_to_metadata.items():
            self._session.execute(update_stmt, (json.dumps(meta), doc_id))

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "ScyllaDBVectorStore":
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store


# ------------------------------------------------------------------
# Filter helpers
# ------------------------------------------------------------------

def _matches_filter(metadata: dict, filter: dict) -> bool:
    """Return True if *metadata* satisfies all equality constraints in *filter*.

    Supports simple ``{"key": value}`` equality filters as used by the
    standard LangChain integration test suite.  Nested ``$and`` / ``$or``
    operators are also handled.
    """
    for key, value in filter.items():
        if key == "$and":
            if not all(_matches_filter(metadata, sub) for sub in value):
                return False
        elif key == "$or":
            if not any(_matches_filter(metadata, sub) for sub in value):
                return False
        elif key == "$not":
            if _matches_filter(metadata, value):
                return False
        else:
            if metadata.get(key) != value:
                return False
    return True
