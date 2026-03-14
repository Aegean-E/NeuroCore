"""
Tests for the Discord Bridge module (modules/discord/).
Mirrors the structure of test_telegram_module.py.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from modules.discord.bridge import DiscordBridge, WEBSOCKETS_AVAILABLE
from modules.discord.node import DiscordInputExecutor, DiscordOutputExecutor
from modules.discord.service import DiscordService


# ---------------------------------------------------------------------------
# DiscordBridge — REST (send_message)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_httpx():
    """Patch httpx.AsyncClient used inside DiscordBridge."""
    with patch("modules.discord.bridge.httpx.AsyncClient") as mock:
        client_instance = AsyncMock()
        mock.return_value = client_instance
        client_instance.post = AsyncMock(return_value=AsyncMock(status_code=200))
        client_instance.is_closed = False
        yield client_instance


@pytest.mark.asyncio
async def test_bridge_send_message_success(mock_httpx):
    bridge = DiscordBridge("token", "999")
    bridge._client = mock_httpx
    result = await bridge.send_message("Hello Discord!")
    assert result is True
    mock_httpx.post.assert_called_once()


@pytest.mark.asyncio
async def test_bridge_send_message_chunking(mock_httpx):
    """Messages longer than 1 900 chars must be split."""
    bridge = DiscordBridge("token", "999")
    bridge._client = mock_httpx
    long_msg = "x" * 4000
    await bridge.send_message(long_msg)
    # 4000 chars → ceil(4000 / 1900) = 3 chunks
    assert mock_httpx.post.call_count == 3


@pytest.mark.asyncio
async def test_bridge_send_message_empty_returns_false():
    bridge = DiscordBridge("token", "999")
    result = await bridge.send_message("")
    assert result is False


@pytest.mark.asyncio
async def test_bridge_send_message_no_channel_returns_false():
    bridge = DiscordBridge("token", "")
    result = await bridge.send_message("hello")
    assert result is False


@pytest.mark.asyncio
async def test_bridge_send_uses_override_channel(mock_httpx):
    """channel_id override in send_message takes precedence over configured default."""
    bridge = DiscordBridge("token", "DEFAULT_CHANNEL")
    bridge._client = mock_httpx
    await bridge.send_message("hi", channel_id="OVERRIDE")
    _, kwargs = mock_httpx.post.call_args
    assert "OVERRIDE" in mock_httpx.post.call_args[0][0]


@pytest.mark.asyncio
async def test_bridge_send_message_http_error_returns_false(mock_httpx):
    mock_httpx.post.side_effect = Exception("Network error")
    bridge = DiscordBridge("token", "999")
    bridge._client = mock_httpx
    result = await bridge.send_message("hello")
    assert result is False


@pytest.mark.asyncio
async def test_bridge_stop_sets_event():
    bridge = DiscordBridge("token", "999")
    import asyncio
    bridge._stop_event = asyncio.Event()
    await bridge.stop()
    assert bridge._stop_event.is_set()


@pytest.mark.asyncio
async def test_bridge_listen_without_websockets(caplog):
    """listen() must log a warning and return immediately when websockets is absent."""
    import logging
    bridge = DiscordBridge("token", "999")
    with patch("modules.discord.bridge.WEBSOCKETS_AVAILABLE", False):
        with caplog.at_level(logging.WARNING):
            await bridge.listen(on_message=lambda m: None, running_check=lambda: True)
    assert any("websockets" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# DiscordInputExecutor / DiscordOutputExecutor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_input_executor_pass_through():
    executor = DiscordInputExecutor()
    data = {"messages": [{"role": "user", "content": "hi"}]}
    result = await executor.receive(data)
    assert result is data


@pytest.mark.asyncio
async def test_input_executor_blocks_repeater():
    executor = DiscordInputExecutor()
    result = await executor.receive({"_repeat_count": 1, "messages": []})
    assert result is None


@pytest.mark.asyncio
async def test_input_executor_send_returns_error_without_messages():
    executor = DiscordInputExecutor()
    result = await executor.send({})
    assert "error" in result


@pytest.mark.asyncio
async def test_output_executor_sends_content():
    with patch(
        "modules.discord.node.ConfigLoader.get_config",
        return_value={"bot_token": "tok", "channel_id": "chan1"},
    ):
        executor = DiscordOutputExecutor()
        executor._bridge_cache = {}
        mock_bridge = AsyncMock()
        executor._bridge_cache["tok_chan1"] = mock_bridge

        await executor.receive({"content": "Hello from flow"})
        mock_bridge.send_message.assert_called_once_with("Hello from flow")


@pytest.mark.asyncio
async def test_output_executor_skips_when_no_content():
    executor = DiscordOutputExecutor()
    result = await executor.receive({"messages": []})
    # No bridge call expected — just passes data through
    assert result == {"messages": []}


@pytest.mark.asyncio
async def test_output_executor_skips_when_no_token():
    with patch(
        "modules.discord.node.ConfigLoader.get_config",
        return_value={"bot_token": "", "channel_id": "chan1"},
    ):
        executor = DiscordOutputExecutor()
        executor._bridge_cache = {}
        # Should not raise — just pass through
        result = await executor.receive({"content": "hi"})
        assert result["content"] == "hi"
        assert executor._bridge_cache == {}


# ---------------------------------------------------------------------------
# DiscordService — process_message
# ---------------------------------------------------------------------------

def _make_discord_msg(content: str, channel_id: str = "ch1", user_id: str = "u1",
                      username: str = "TestUser") -> dict:
    return {
        "channel_id": channel_id,
        "content": content,
        "author": {"id": user_id, "username": username, "bot": False},
    }


@pytest.mark.asyncio
async def test_service_ignores_bot_messages():
    service = DiscordService()
    service.bridge = AsyncMock()
    msg = {
        "channel_id": "ch1",
        "content": "hello",
        "author": {"id": "bot1", "username": "SomeBot", "bot": True},
    }
    await service.process_message(msg)
    service.bridge.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_service_respects_channel_filter():
    """Messages from channels other than the configured one are ignored."""
    service = DiscordService()
    service.bridge = AsyncMock()
    with patch(
        "modules.discord.service.ConfigLoader.get_config",
        return_value={"bot_token": "t", "channel_id": "ALLOWED"},
    ):
        msg = _make_discord_msg("hello", channel_id="OTHER")
        await service.process_message(msg)
    service.bridge.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_service_help_command():
    service = DiscordService()
    service.bridge = AsyncMock()
    with patch(
        "modules.discord.service.ConfigLoader.get_config",
        return_value={"bot_token": "t", "channel_id": "ch1"},
    ):
        await service.process_message(_make_discord_msg("/help"))
    args, _ = service.bridge.send_message.call_args
    assert "NeuroCore Discord Bridge Commands" in args[0]


@pytest.mark.asyncio
async def test_service_new_session_command():
    service = DiscordService()
    service.bridge = AsyncMock()
    service.session_map = {}
    with patch(
        "modules.discord.service.ConfigLoader.get_config",
        return_value={"bot_token": "t", "channel_id": "ch1"},
    ), patch("modules.discord.service.session_manager") as mock_sm:
        mock_sm.create_session.return_value = {"id": "new-sess-123"}
        await service.process_message(_make_discord_msg("/new_session"))
    assert service.session_map.get("ch1:u1") == "new-sess-123"
    service.bridge.send_message.assert_called_once()
    assert "New session" in service.bridge.send_message.call_args[0][0]


@pytest.mark.asyncio
async def test_service_delete_session_command():
    service = DiscordService()
    service.bridge = AsyncMock()
    service.session_map = {"ch1:u1": "sess-to-del"}
    with patch(
        "modules.discord.service.ConfigLoader.get_config",
        return_value={"bot_token": "t", "channel_id": "ch1"},
    ), patch("modules.discord.service.session_manager"):
        await service.process_message(_make_discord_msg("/delete_session"))
    assert "ch1:u1" not in service.session_map
    assert "Session deleted" in service.bridge.send_message.call_args[0][0]


@pytest.mark.asyncio
async def test_service_no_active_flow():
    service = DiscordService()
    service.bridge = AsyncMock()
    service.session_map = {}
    with patch(
        "modules.discord.service.ConfigLoader.get_config",
        return_value={"bot_token": "t", "channel_id": "ch1"},
    ), patch("modules.discord.service.session_manager") as mock_sm, \
       patch("modules.discord.service.settings") as mock_settings:
        mock_sm.create_session.return_value = {"id": "s1"}
        mock_sm.get_session.return_value = {"history": []}
        mock_settings.get.return_value = []  # no active flows
        await service.process_message(_make_discord_msg("hello"))
    args = service.bridge.send_message.call_args[0][0]
    assert "No active AI Flow" in args


@pytest.mark.asyncio
async def test_service_flow_execution():
    service = DiscordService()
    service.bridge = AsyncMock()
    service.session_map = {}
    with patch(
        "modules.discord.service.ConfigLoader.get_config",
        return_value={"bot_token": "t", "channel_id": "ch1"},
    ), patch("modules.discord.service.session_manager") as mock_sm, \
       patch("modules.discord.service.settings") as mock_settings, \
       patch("modules.discord.service.FlowRunner") as MockRunner:
        mock_settings.get.return_value = ["flow-abc"]
        mock_sm.create_session.return_value = {"id": "s1"}
        mock_sm.get_session.return_value = {"history": []}
        runner_instance = MockRunner.return_value
        runner_instance.run = AsyncMock(return_value={"content": "AI response"})

        await service.process_message(_make_discord_msg("Tell me something"))

    MockRunner.assert_called_with(flow_id="flow-abc")
    runner_instance.run.assert_called_once()
    sent_text = service.bridge.send_message.call_args[0][0]
    assert "AI response" in sent_text
    assert mock_sm.add_message.called


@pytest.mark.asyncio
async def test_service_flow_error_handled():
    """A flow execution error must send an error reply, not crash."""
    service = DiscordService()
    service.bridge = AsyncMock()
    service.session_map = {}
    with patch(
        "modules.discord.service.ConfigLoader.get_config",
        return_value={"bot_token": "t", "channel_id": "ch1"},
    ), patch("modules.discord.service.session_manager") as mock_sm, \
       patch("modules.discord.service.settings") as mock_settings, \
       patch("modules.discord.service.FlowRunner") as MockRunner:
        mock_settings.get.return_value = ["flow-abc"]
        mock_sm.create_session.return_value = {"id": "s1"}
        mock_sm.get_session.return_value = {"history": []}
        runner_instance = MockRunner.return_value
        runner_instance.run = AsyncMock(side_effect=RuntimeError("boom"))

        await service.process_message(_make_discord_msg("crash me"))

    sent_text = service.bridge.send_message.call_args[0][0]
    assert "Internal Error" in sent_text or "❌" in sent_text


@pytest.mark.asyncio
async def test_service_input_source_is_discord():
    """The _input_source key passed to FlowRunner must be 'discord'."""
    service = DiscordService()
    service.bridge = AsyncMock()
    service.session_map = {}
    captured_data = {}
    with patch(
        "modules.discord.service.ConfigLoader.get_config",
        return_value={"bot_token": "t", "channel_id": "ch1"},
    ), patch("modules.discord.service.session_manager") as mock_sm, \
       patch("modules.discord.service.settings") as mock_settings, \
       patch("modules.discord.service.FlowRunner") as MockRunner:
        mock_settings.get.return_value = ["flow-abc"]
        mock_sm.create_session.return_value = {"id": "s1"}
        mock_sm.get_session.return_value = {"history": []}

        async def capture_run(data):
            captured_data.update(data)
            return {"content": "ok"}

        MockRunner.return_value.run = capture_run
        await service.process_message(_make_discord_msg("hello"))

    assert captured_data.get("_input_source") == "discord"
