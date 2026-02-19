import pytest
import asyncio
from core.llm import LLMBridge

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
    
    bridge = LLMBridge(base_url="http://localhost:1234/v1")
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
    
    bridge = LLMBridge(base_url="http://localhost:1234/v1")
    result = await bridge.chat_completion([{"role": "user", "content": "hi"}])
    
    assert "error" in result

@pytest.mark.asyncio
async def test_get_models_success(httpx_mock):
    mock_response = {
        "data": [
            {"id": "model-1"},
            {"id": "model-2"}
        ]
    }
    httpx_mock.add_response(
        url="http://localhost:1234/v1/models",
        json=mock_response
    )
    bridge = LLMBridge(base_url="http://localhost:1234/v1")
    result = await bridge.get_models()
    assert "data" in result
    assert len(result["data"]) == 2

@pytest.mark.asyncio
async def test_get_models_error(httpx_mock):
    httpx_mock.add_response(url="http://localhost:1234/v1/models", status_code=500)
    bridge = LLMBridge(base_url="http://localhost:1234/v1")
    result = await bridge.get_models()
    assert "error" in result
