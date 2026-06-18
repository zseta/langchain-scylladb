"""Integration tests for ScyllaDBChatMessageHistory (local testcontainers)."""
from __future__ import annotations

import pytest
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy
from langchain_core.messages import AIMessage, HumanMessage

from langchain_scylladb.chat_message_histories import ScyllaDBChatMessageHistory


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
    # Drop the test table after each test
    s.execute(f"DROP TABLE IF EXISTS {info['keyspace']}.chat_history_test")
    cluster.shutdown()


def _make_history(session, keyspace, session_id: str, **kwargs) -> ScyllaDBChatMessageHistory:
    return ScyllaDBChatMessageHistory(
        session=session,
        session_id=session_id,
        keyspace=keyspace,
        table_name="chat_history_test",
        **kwargs,
    )


def test_roundtrip_human_and_ai_messages(session) -> None:
    db_session, keyspace = session
    history = _make_history(db_session, keyspace, "s1")

    history.add_messages([HumanMessage(content="hi"), AIMessage(content="hello")])
    msgs = history.messages

    assert len(msgs) == 2
    assert msgs[0].content == "hi"
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[1].content == "hello"
    assert isinstance(msgs[1], AIMessage)


def test_multi_session_isolation(session) -> None:
    db_session, keyspace = session
    hist_a = _make_history(db_session, keyspace, "session-a")
    hist_b = _make_history(db_session, keyspace, "session-b")

    hist_a.add_messages([HumanMessage(content="for a")])
    hist_b.add_messages([HumanMessage(content="for b")])

    assert len(hist_a.messages) == 1
    assert hist_a.messages[0].content == "for a"
    assert len(hist_b.messages) == 1
    assert hist_b.messages[0].content == "for b"


def test_history_size_returns_only_last_n(session) -> None:
    db_session, keyspace = session
    history = _make_history(db_session, keyspace, "s-limit", history_size=1)

    history.add_messages([HumanMessage(content="first"), HumanMessage(content="last")])
    msgs = history.messages

    assert len(msgs) == 1
    assert msgs[0].content == "last"


def test_clear_removes_messages_for_session_only(session) -> None:
    db_session, keyspace = session
    hist_a = _make_history(db_session, keyspace, "clear-a")
    hist_b = _make_history(db_session, keyspace, "clear-b")

    hist_a.add_messages([HumanMessage(content="a")])
    hist_b.add_messages([HumanMessage(content="b")])

    hist_a.clear()

    assert hist_a.messages == []
    assert len(hist_b.messages) == 1


@pytest.mark.asyncio
async def test_async_add_and_get_messages(session) -> None:
    db_session, keyspace = session
    history = _make_history(db_session, keyspace, "async-session")

    await history.aadd_messages([HumanMessage(content="async hi")])
    msgs = await history.aget_messages()

    assert len(msgs) == 1
    assert msgs[0].content == "async hi"


@pytest.mark.asyncio
async def test_aclear(session) -> None:
    db_session, keyspace = session
    history = _make_history(db_session, keyspace, "async-clear")

    await history.aadd_messages([HumanMessage(content="to be cleared")])
    await history.aclear()

    msgs = await history.aget_messages()
    assert msgs == []
