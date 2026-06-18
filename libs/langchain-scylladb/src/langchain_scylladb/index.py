"""ScyllaDB schema / index utilities.

Standalone helpers for creating, inspecting, and dropping vector indexes
and secondary indexes.  These can be used in scripts and migrations
independently of ``ScyllaDBVectorStore``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from cassandra.cluster import Session
from cassandra.query import SimpleStatement

_INDEXES_SELECT = SimpleStatement(
    "SELECT index_name FROM system_schema.indexes "
    "WHERE keyspace_name = %s AND table_name = %s"
)


def create_vector_index(
    session: Session,
    keyspace: str,
    table: str,
    column: str,
    similarity: str = "COSINE",
    options: Optional[Dict[str, Any]] = None,
    index_name: Optional[str] = None,
) -> None:
    """Create an HNSW vector index on *column* in *keyspace*.*table*.

    Args:
        session: An open ``cassandra.cluster.Session``.
        keyspace: Target keyspace.
        table: Target table.
        column: Column of type ``vector<float, N>`` to index.
        similarity: Similarity function — ``"COSINE"`` (default),
            ``"DOT_PRODUCT"``, or ``"EUCLIDEAN"``.
        options: Additional ``WITH OPTIONS`` key/value pairs.
        index_name: Index name; defaults to ``{table}_{column}_idx``.
    """
    if index_name is None:
        index_name = f"{table}_{column}_idx"
    opts: Dict[str, Any] = {"similarity_function": similarity}
    if options:
        opts.update(options)
    opts_str = ", ".join(f"'{k}': '{v}'" for k, v in opts.items())
    session.execute(
        f"""
        CREATE CUSTOM INDEX IF NOT EXISTS {index_name}
        ON {keyspace}.{table}({column})
        USING 'vector_index'
        WITH OPTIONS = {{{opts_str}}}
        """
    )


def drop_vector_index(session: Session, keyspace: str, index_name: str) -> None:
    """Drop a vector index.

    Args:
        session: An open ``cassandra.cluster.Session``.
        keyspace: Keyspace that owns the index.
        index_name: Name of the index to drop.
    """
    session.execute(f"DROP INDEX IF EXISTS {keyspace}.{index_name}")


def create_secondary_index(
    session: Session,
    keyspace: str,
    table: str,
    column: str,
    index_name: Optional[str] = None,
) -> None:
    """Create a regular secondary index on *column*.

    Args:
        session: An open ``cassandra.cluster.Session``.
        keyspace: Target keyspace.
        table: Target table.
        column: Column to index.
        index_name: Index name; defaults to ``{table}_{column}_idx``.
    """
    if index_name is None:
        index_name = f"{table}_{column}_idx"
    session.execute(
        f"CREATE INDEX IF NOT EXISTS {index_name} ON {keyspace}.{table}({column})"
    )


def wait_for_index(
    session: Session,
    keyspace: str,
    table: str,
    index_name: str,
    timeout: float = 60.0,
    poll_interval: float = 1.0,
) -> None:
    """Poll ``system_schema.indexes`` until *index_name* appears.

    Args:
        session: An open ``cassandra.cluster.Session``.
        keyspace: Target keyspace.
        table: Target table.
        index_name: Name of the index to wait for.
        timeout: Maximum seconds to wait before raising ``TimeoutError``.
        poll_interval: Seconds between polls.

    Raises:
        TimeoutError: If the index does not appear within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = session.execute(_INDEXES_SELECT, (keyspace, table))
        for row in rows:
            if row.index_name == index_name:
                return
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Index {index_name!r} on {keyspace}.{table} did not appear "
        f"within {timeout}s"
    )


def list_indexes(session: Session, keyspace: str, table: str) -> List[str]:
    """Return a list of index names for *keyspace*.*table*.

    Args:
        session: An open ``cassandra.cluster.Session``.
        keyspace: Target keyspace.
        table: Target table.
    """
    rows = session.execute(_INDEXES_SELECT, (keyspace, table))
    return [row.index_name for row in rows]
