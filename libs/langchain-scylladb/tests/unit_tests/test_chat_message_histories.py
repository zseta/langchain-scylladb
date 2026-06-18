"""Unit tests for ScyllaDBChatMessageHistory."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict

from langchain_scylladb.chat_message_histories import ScyllaDBChatMessageHistory


def _make_history(**kwargs) -> tuple[ScyllaDBChatMessageHistory, MagicMock]:
    fake_session = MagicMock()
    with patch.object(ScyllaDBChatMessageHistory, "_setup_schema"):
        history = ScyllaDBChatMessageHistory(
            session=fake_session,
            session_id="test-session",
            **kwargs,
        )
    return history, fake_session


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------


def test_setup_schema_called_on_init() -> None:
    fake_session = MagicMock()
    with patch.object(ScyllaDBChatMessageHistory, "_setup_schema") as mock_setup:
        ScyllaDBChatMessageHistory(session=fake_session, session_id="s1")
    mock_setup.assert_called_once()


# ---------------------------------------------------------------------------
# messages property
# ---------------------------------------------------------------------------


def test_messages_deserializes_stored_json() -> None:
    history, fake_session = _make_history()
    msgs = [HumanMessage(content="hello"), AIMessage(content="world")]
    msg_dicts = [json.dumps(d) for d in messages_to_dict(msgs)]

    row_0 = MagicMock()
    row_0.message = msg_dicts[0]
    row_1 = MagicMock()
    row_1.message = msg_dicts[1]

    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.return_value = [row_0, row_1]

    result = history.messages
    assert len(result) == 2
    assert result[0].content == "hello"
    assert result[1].content == "world"


def test_history_size_adds_limit_to_query() -> None:
    history, fake_session = _make_history(history_size=5)
    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.return_value = []

    history.messages

    prepared_query: str = fake_session.prepare.call_args[0][0]
    assert "LIMIT 5" in prepared_query
    assert "DESC" in prepared_query


def test_messages_without_history_size_fetches_all() -> None:
    history, fake_session = _make_history()
    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.return_value = []

    history.messages

    prepared_query: str = fake_session.prepare.call_args[0][0]
    assert "LIMIT" not in prepared_query


# ---------------------------------------------------------------------------
# add_messages
# ---------------------------------------------------------------------------


def test_add_messages_batches_all_inserts_in_single_execute() -> None:
    history, fake_session = _make_history()
    fake_session.prepare.return_value = MagicMock()

    msgs = [HumanMessage(content="a"), AIMessage(content="b")]
    history.add_messages(msgs)

    # prepare called once for insert, execute called once for the batch
    assert fake_session.execute.call_count == 1


def test_add_messages_empty_list_does_nothing() -> None:
    history, fake_session = _make_history()
    history.add_messages([])
    fake_session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_issues_delete_with_session_id() -> None:
    history, fake_session = _make_history()
    fake_session.prepare.return_value = MagicMock()

    history.clear()

    fake_session.prepare.assert_called_once()
    prepared_query: str = fake_session.prepare.call_args[0][0]
    assert "DELETE" in prepared_query
    fake_session.execute.assert_called_once_with(fake_session.prepare.return_value, ("test-session",))


# ---------------------------------------------------------------------------
# Async wrappers delegate to sync implementations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_messages_returns_same_as_messages() -> None:
    history, fake_session = _make_history()
    fake_session.prepare.return_value = MagicMock()
    fake_session.execute.return_value = []

    assert await history.aget_messages() == history.messages


@pytest.mark.asyncio
async def test_aclear_delegates_to_clear() -> None:
    history, fake_session = _make_history()
    fake_session.prepare.return_value = MagicMock()
    await history.aclear()
    fake_session.execute.assert_called()
