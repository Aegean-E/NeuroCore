import pytest
import asyncio
from core.llm import LMStudioBridge

@pytest.mark.asyncio
async def test_chat_completion_success(httpx_mock):
    mock_response = {
        "choices": [
            {"message": {"content": "Test response"}}
        ]
    }
    httpx_mock.add_response(
        url="http://localhost:1234/v1/chat/completions",
        json=mock_response,
        method="POST"
    )
    
    bridge = LMStudioBridge()
    result = await bridge.chat_completion([{"role": "user", "content": "hi"}])
    
    assert "choices" in result
    assert result["choices"][0]["message"]["content"] == "Test response"

@pytest.mark.asyncio
async def test_chat_completion_error(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:1234/v1/chat/completions",
        status_code=500,
        method="POST"
    )
    
    bridge = LMStudioBridge()
    result = await bridge.chat_completion([{"role": "user", "content": "hi"}])
    
    assert "error" in result
