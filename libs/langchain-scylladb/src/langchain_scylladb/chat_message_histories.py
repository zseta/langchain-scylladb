"""ScyllaDB-backed chat message history."""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Sequence
from uuid import uuid1

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from cassandra.cluster import Session
from cassandra.query import BatchStatement, BatchType

logger = logging.getLogger(__name__)


class ScyllaDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in ScyllaDB.

    CQL schema::

        CREATE TABLE {keyspace}.{table} (
            session_id  TEXT,
            message_id  TIMEUUID,
            message     TEXT,
            PRIMARY KEY (session_id, message_id)
        ) WITH CLUSTERING ORDER BY (message_id ASC);

    ``session_id`` is the partition key so all messages for one session are
    co-located.  ``TIMEUUID`` provides monotonic, time-based ordering at the
    storage layer without a separate sequence column.

    Args:
        session: An open ``cassandra.cluster.Session``.
        session_id: Identifier for the conversation session.
        keyspace: ScyllaDB keyspace (must already exist).
        table_name: Table name within the keyspace.
        history_size: If set, only the most recent *history_size* messages are
            returned by the ``messages`` property.
    """

    def __init__(
        self,
        session: Session,
        *,
        session_id: str,
        keyspace: str = "langchain_scylladb",
        table_name: str = "chat_history",
        history_size: Optional[int] = None,
    ) -> None:
        self._db_session = session
        self._session_id = session_id
        self._keyspace = keyspace
        self._table_name = table_name
        self._history_size = history_size
        self._setup_schema()

    def _setup_schema(self) -> None:
        self._db_session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._table_name} (
                session_id  TEXT,
                message_id  TIMEUUID,
                message     TEXT,
                PRIMARY KEY (session_id, message_id)
            ) WITH CLUSTERING ORDER BY (message_id ASC)
            """
        )

    @property
    def messages(self) -> List[BaseMessage]:
        if self._history_size is not None:
            # DESC + LIMIT to get the N most recent, then reverse for chronological order
            stmt = self._db_session.prepare(
                f"SELECT message FROM {self._keyspace}.{self._table_name} "
                f"WHERE session_id = ? ORDER BY message_id DESC LIMIT {self._history_size}"
            )
            rows = list(self._db_session.execute(stmt, (self._session_id,)))
            rows.reverse()
        else:
            stmt = self._db_session.prepare(
                f"SELECT message FROM {self._keyspace}.{self._table_name} "
                f"WHERE session_id = ?"
            )
            rows = list(self._db_session.execute(stmt, (self._session_id,)))

        dicts = [json.loads(row.message) for row in rows]
        return messages_from_dict(dicts)

    async def aget_messages(self) -> List[BaseMessage]:
        return self.messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        insert_stmt = self._db_session.prepare(
            f"INSERT INTO {self._keyspace}.{self._table_name} "
            f"(session_id, message_id, message) VALUES (?, ?, ?)"
        )
        batch = BatchStatement(batch_type=BatchType.UNLOGGED)
        for msg in messages:
            message_id = uuid1()
            message_json = json.dumps(messages_to_dict([msg])[0])
            batch.add(insert_stmt, (self._session_id, message_id, message_json))
        self._db_session.execute(batch)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.add_messages(messages)

    def clear(self) -> None:
        stmt = self._db_session.prepare(
            f"DELETE FROM {self._keyspace}.{self._table_name} WHERE session_id = ?"
        )
        self._db_session.execute(stmt, (self._session_id,))

    async def aclear(self) -> None:
        self.clear()
