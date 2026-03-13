"""
Tests for modules/telegram/node.py — TelegramInputExecutor, TelegramOutputExecutor
"""
import pytest
from unittest.mock import MagicMock, patch
from modules.telegram.node import TelegramInputExecutor, TelegramOutputExecutor, get_executor_class


# ---------------------------------------------------------------------------
# TelegramInputExecutor
# ---------------------------------------------------------------------------

class TestTelegramInput:

    @pytest.mark.asyncio
    async def test_normal_input_passes_through(self):
        """Normal input (no repeat) should pass through unchanged."""
        executor = TelegramInputExecutor()
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}
        result = await executor.receive(input_data)
        assert result is input_data

    @pytest.mark.asyncio
    async def test_repeat_input_returns_none(self):
        """Input with _repeat_count > 0 should return None (suppress repeater trigger)."""
        executor = TelegramInputExecutor()
        result = await executor.receive({"_repeat_count": 1, "messages": []})
        assert result is None

    @pytest.mark.asyncio
    async def test_repeat_count_zero_passes_through(self):
        """Input with _repeat_count == 0 should pass through normally."""
        executor = TelegramInputExecutor()
        input_data = {"_repeat_count": 0, "messages": [{"role": "user", "content": "Hi"}]}
        result = await executor.receive(input_data)
        assert result is input_data

    @pytest.mark.asyncio
    async def test_send_with_messages_passes_through(self):
        """send() with 'messages' key should pass through."""
        executor = TelegramInputExecutor()
        data = {"messages": [{"role": "user", "content": "Hi"}]}
        result = await executor.send(data)
        assert result is data

    @pytest.mark.asyncio
    async def test_send_without_messages_returns_error(self):
        """send() without 'messages' or 'text' key should return an error dict."""
        executor = TelegramInputExecutor()
        result = await executor.send({"other": "data"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_send_repeat_suppressed(self):
        """send() with _repeat_count > 0 should pass through without error."""
        executor = TelegramInputExecutor()
        data = {"_repeat_count": 2, "other": "data"}
        result = await executor.send(data)
        # Should not produce an error for repeater-triggered calls
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_send_with_text_key_passes_through(self):
        """send() with 'text' key (Telegram update format) should pass through."""
        executor = TelegramInputExecutor()
        data = {"text": "Hello from Telegram"}
        result = await executor.send(data)
        assert result is data


# ---------------------------------------------------------------------------
# TelegramOutputExecutor
# ---------------------------------------------------------------------------

class TestTelegramOutput:

    @pytest.mark.asyncio
    async def test_no_content_passthrough(self):
        """Input without 'content' key should be returned unchanged (no send attempt)."""
        executor = TelegramOutputExecutor()
        input_data = {"messages": []}
        result = await executor.receive(input_data)
        assert result is input_data

    @pytest.mark.asyncio
    async def test_none_input_passthrough(self):
        """None input should be returned as-is."""
        executor = TelegramOutputExecutor()
        result = await executor.receive(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_sends_message_when_credentials_present(self):
        """When bot_token and chat_id are configured, TelegramBridge.send_message should be called."""
        executor = TelegramOutputExecutor()
        input_data = {"content": "Hello, Telegram!"}

        mock_config = {"bot_token": "test_token", "chat_id": "12345"}
        mock_bridge = MagicMock()

        with patch("modules.telegram.node.ConfigLoader.get_config", return_value=mock_config), \
             patch("modules.telegram.node.TelegramBridge", return_value=mock_bridge):
            result = await executor.receive(input_data)

        assert mock_bridge.send_message.called  # Called at least once
        assert result is input_data

    @pytest.mark.asyncio
    async def test_skips_send_when_no_token(self):
        """When bot_token is missing, no send attempt should be made."""
        executor = TelegramOutputExecutor()
        input_data = {"content": "Hello"}

        mock_config = {"bot_token": None, "chat_id": "12345"}

        with patch("modules.telegram.node.ConfigLoader.get_config", return_value=mock_config), \
             patch("modules.telegram.node.TelegramBridge") as MockBridge:
            result = await executor.receive(input_data)

        MockBridge.assert_not_called()
        assert result is input_data

    @pytest.mark.asyncio
    async def test_skips_send_when_no_chat_id(self):
        """When chat_id is missing, no send attempt should be made."""
        executor = TelegramOutputExecutor()
        input_data = {"content": "Hello"}

        mock_config = {"bot_token": "test_token", "chat_id": None}

        with patch("modules.telegram.node.ConfigLoader.get_config", return_value=mock_config), \
             patch("modules.telegram.node.TelegramBridge") as MockBridge:
            result = await executor.receive(input_data)

        MockBridge.assert_not_called()
        assert result is input_data

    @pytest.mark.asyncio
    async def test_send_passthrough(self):
        """send() should return processed_data unchanged."""
        executor = TelegramOutputExecutor()
        data = {"content": "Done"}
        result = await executor.send(data)
        assert result is data


# ---------------------------------------------------------------------------
# get_executor_class dispatcher
# ---------------------------------------------------------------------------

class TestGetExecutorClass:

    @pytest.mark.asyncio
    async def test_telegram_input_dispatcher(self):
        cls = await get_executor_class("telegram_input")
        assert cls is TelegramInputExecutor

    @pytest.mark.asyncio
    async def test_telegram_output_dispatcher(self):
        cls = await get_executor_class("telegram_output")
        assert cls is TelegramOutputExecutor

    @pytest.mark.asyncio
    async def test_unknown_returns_none(self):
        cls = await get_executor_class("unknown")
        assert cls is None
