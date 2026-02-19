import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from modules.memory.node import MemoryRecallExecutor, MemorySaveExecutor

@pytest.fixture
def mock_store():
    with patch("modules.memory.node.memory_store") as ms:
        yield ms

@pytest.fixture
def mock_llm_bridge():
    with patch("modules.memory.node.LLMBridge") as MockBridge:
        instance = MockBridge.return_value
        instance.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        yield instance

@pytest.mark.asyncio
async def test_memory_recall_success(mock_store, mock_llm_bridge):
    """Test that relevant memories are injected into the message history."""
    executor = MemoryRecallExecutor()
    
    # Setup mock store results
    mock_store.search.return_value = [
        {"text": "User likes apples", "score": 0.9},
        {"text": "User lives in NYC", "score": 0.8}
    ]
    
    input_data = {
        "messages": [
            {"role": "user", "content": "What do I like?"}
        ]
    }
    
    result = await executor.receive(input_data, config={"limit": 2})
    
    # Verify embedding was requested
    mock_llm_bridge.get_embedding.assert_called_with("What do I like?")
    
    # Verify messages structure
    messages = result["messages"]
    assert len(messages) == 2  # System injection + User message
    assert messages[0]["role"] == "system"
    assert "User likes apples" in messages[0]["content"]
    assert messages[1]["role"] == "user"

@pytest.mark.asyncio
async def test_memory_recall_no_results(mock_store, mock_llm_bridge):
    """Test that nothing changes if no memories are found."""
    executor = MemoryRecallExecutor()
    mock_store.search.return_value = []
    
    input_data = {"messages": [{"role": "user", "content": "Hi"}]}
    result = await executor.receive(input_data)
    
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"

@pytest.mark.asyncio
async def test_memory_save_assistant_content(mock_store, mock_llm_bridge):
    """Test saving content from Chat Output (Assistant)."""
    executor = MemorySaveExecutor()
    
    # Mock the background task execution by calling the internal method directly
    # or by ensuring the receive method triggers the logic.
    # Since receive uses asyncio.create_task, we can't easily await it in a unit test 
    # without mocking create_task or the internal _save_background.
    
    with patch.object(executor, '_save_background', new_callable=AsyncMock) as mock_save_bg:
        input_data = {"content": "I am an AI."}
        await executor.receive(input_data)
        
        mock_save_bg.assert_called_once()
        args, kwargs = mock_save_bg.call_args
        assert args[0] == "I am an AI."
        assert kwargs["subject"] == "Assistant"

@pytest.mark.asyncio
async def test_memory_save_openai_response(mock_store, mock_llm_bridge):
    """Test saving content from raw OpenAI-style response (LLM Core)."""
    executor = MemorySaveExecutor()
    
    with patch.object(executor, '_save_background', new_callable=AsyncMock) as mock_save_bg:
        input_data = {
            "choices": [
                {"message": {"content": "Generated thought."}}
            ]
        }
        await executor.receive(input_data)
        
        mock_save_bg.assert_called_once()
        args, kwargs = mock_save_bg.call_args
        assert args[0] == "Generated thought."
        assert kwargs["subject"] == "Assistant"

@pytest.mark.asyncio
async def test_memory_save_user_message(mock_store, mock_llm_bridge):
    """Test saving content from Chat Input (User)."""
    executor = MemorySaveExecutor()
    
    with patch.object(executor, '_save_background', new_callable=AsyncMock) as mock_save_bg:
        input_data = {
            "messages": [
                {"role": "user", "content": "My name is John."}
            ]
        }
        await executor.receive(input_data)
        
        mock_save_bg.assert_called_once()
        args, kwargs = mock_save_bg.call_args
        assert args[0] == "My name is John."
        assert kwargs["subject"] == "User"

@pytest.mark.asyncio
async def test_memory_save_short_text_ignored(mock_store):
    """Test that short text is ignored."""
    executor = MemorySaveExecutor()
    with patch.object(executor, '_save_background', new_callable=AsyncMock) as mock_save_bg:
        await executor.receive({"content": "H"}) # Length 1, should be ignored (threshold is > 1)
        mock_save_bg.assert_not_called()

@pytest.mark.asyncio
async def test_memory_recall_min_score_filtering(mock_store, mock_llm_bridge):
    """Test that results below min_score are filtered out."""
    executor = MemoryRecallExecutor()
    
    mock_store.search.return_value = [
        {"text": "High score", "score": 0.9},
        {"text": "Low score", "score": 0.4}
    ]
    
    input_data = {"messages": [{"role": "user", "content": "Query"}]}
    
    # Set min_score to 0.5
    result = await executor.receive(input_data, config={"min_score": 0.5})
    
    messages = result["messages"]
    system_content = messages[0]["content"]
    
    assert "High score" in system_content
    assert "Low score" not in system_content

@pytest.mark.asyncio
async def test_memory_save_config_passing(mock_store):
    """Test that config values (type, confidence) are passed to save."""
    executor = MemorySaveExecutor()
    with patch.object(executor, '_save_background', new_callable=AsyncMock) as mock_save_bg:
        input_data = {"content": "Important fact"}
        config = {"mem_type": "RULE", "confidence": 0.95}
        
        await executor.receive(input_data, config)
        
        mock_save_bg.assert_called_once()
        _, kwargs = mock_save_bg.call_args
        assert kwargs["mem_type"] == "RULE"
        assert kwargs["confidence"] == 0.95

@pytest.mark.asyncio
async def test_save_background_logic(mock_store):
    """Test the _save_background method logic."""
    executor = MemorySaveExecutor()
    
    # Mock Arbiter
    executor.arbiter = MagicMock()
    
    # Mock LLMBridge
    with patch("modules.memory.node.LLMBridge") as MockBridge:
        bridge_instance = MockBridge.return_value
        bridge_instance.get_embedding = AsyncMock(return_value=[0.1, 0.2])
        
        await executor._save_background("Text", subject="User", mem_type="FACT", confidence=0.9)
        
        bridge_instance.get_embedding.assert_called_with("Text")
        executor.arbiter.consider.assert_called_once()
        _, kwargs = executor.arbiter.consider.call_args
        assert kwargs["text"] == "Text"
        assert kwargs["embedding"] == [0.1, 0.2]
        assert kwargs["subject"] == "User"
        assert kwargs["mem_type"] == "FACT"
        assert kwargs["confidence"] == 0.9

@pytest.mark.asyncio
async def test_save_background_embedding_failure(mock_store):
    """Test that save is aborted if embedding generation fails."""
    executor = MemorySaveExecutor()
    executor.arbiter = MagicMock()
    
    with patch("modules.memory.node.LLMBridge") as MockBridge:
        bridge_instance = MockBridge.return_value
        # Simulate failure (None)
        bridge_instance.get_embedding = AsyncMock(return_value=None)
        
        await executor._save_background("Text")
        
        # Arbiter should NOT be called
        executor.arbiter.consider.assert_not_called()

@pytest.mark.asyncio
async def test_memory_recall_no_user_message(mock_store, mock_llm_bridge):
    """Test that recall does nothing if no user message is found."""
    executor = MemoryRecallExecutor()
    
    # Case 1: Empty messages
    res1 = await executor.receive({"messages": []})
    assert res1 == {"messages": []}
    mock_llm_bridge.get_embedding.assert_not_called()
    
    # Case 2: Only system/assistant messages
    input_data = {
        "messages": [
            {"role": "system", "content": "Sys"},
            {"role": "assistant", "content": "Hi"}
        ]
    }
    res2 = await executor.receive(input_data)
    assert res2 == input_data
    mock_llm_bridge.get_embedding.assert_not_called()