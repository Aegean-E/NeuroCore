import pytest
import os
import asyncio
from unittest.mock import patch, AsyncMock
from core.flow_runner import FlowRunner
from modules.memory.backend import MemoryStore

TEST_DB = "test_integration_mem.sqlite3"

@pytest.fixture
def temp_memory_store():
    if os.path.exists(TEST_DB): os.remove(TEST_DB)
    store = MemoryStore(db_path=TEST_DB)
    yield store
    # Cleanup
    if os.path.exists(TEST_DB): os.remove(TEST_DB)
    if os.path.exists(TEST_DB.replace(".sqlite3", ".faiss")): 
        os.remove(TEST_DB.replace(".sqlite3", ".faiss"))

@pytest.mark.asyncio
async def test_memory_flow_integration(temp_memory_store):
    """
    Integration test for a flow with Memory Recall -> LLM -> Memory Save.
    Verifies that data saved in one turn is available in the next.
    """
    # Patch settings directly to ensure all components use the mock URL
    from core.settings import settings as global_settings
    
    with patch.dict(global_settings.settings, {"llm_api_url": "http://localhost:1234/v1", "embedding_api_url": "http://localhost:1234/v1"}), \
         patch("modules.memory.node.memory_store", temp_memory_store), \
         patch("modules.memory.backend.memory_store", temp_memory_store), \
         patch("modules.memory.consolidation.memory_store", temp_memory_store):
        
        # Mock LLMBridge globally to avoid real API calls from any module
        with patch("core.llm.LLMBridge") as MockBridge:
            
            # Setup LLM Mock
            bridge_instance = MockBridge.return_value
            # Return a constant embedding so search always matches
            bridge_instance.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            # Setup chat completion
            bridge_instance.chat_completion = AsyncMock(return_value={
                    "choices": [{"message": {"content": '["My secret code is 1234"]'}}]
            })

            # Define a linear flow
            flow_def = {
                "id": "integration-flow",
                "nodes": [
                    {"id": "n1", "moduleId": "chat", "nodeTypeId": "chat_input", "name": "Input"},
                    {"id": "n2", "moduleId": "memory", "nodeTypeId": "memory_recall", "name": "Recall"},
                    {"id": "n3", "moduleId": "llm_module", "nodeTypeId": "llm_module", "name": "LLM"},
                    {"id": "n4", "moduleId": "memory", "nodeTypeId": "memory_save", "name": "Save"},
                    {"id": "n5", "moduleId": "chat", "nodeTypeId": "chat_output", "name": "Output"}
                ],
                "connections": [
                        {"from": "n1", "to": "n4"},
                        {"from": "n4", "to": "n2"},
                        {"from": "n2", "to": "n3"},
                        {"from": "n3", "to": "n5"}
                ]
            }
            
            # Mock FlowManager to return our test flow
            with patch("core.flow_runner.flow_manager") as mock_fm:
                mock_fm.get_flow.return_value = flow_def
                
                runner = FlowRunner("integration-flow")
                
                # --- Turn 1: Save Memory ---
                input_data = {"messages": [{"role": "user", "content": "My secret code is 1234."}]}
                
                # We need to ensure the background save task completes.
                # We patch asyncio.create_task to await the coroutine immediately for testing purposes.
                tasks = []
                original_create_task = asyncio.create_task
                def capture_task(coro):
                    t = original_create_task(coro)
                    tasks.append(t)
                    return t
                
                with patch("asyncio.create_task", side_effect=capture_task):
                    await runner.run(input_data)
                    if tasks:
                        await asyncio.gather(*tasks)
                
                # Verify memory was saved to store
                stats = temp_memory_store.get_memory_stats()
                assert stats['total'] == 1
                
                # --- Turn 2: Recall Memory ---
                input_data_2 = {"messages": [{"role": "user", "content": "What is my code?"}]}
                await runner.run(input_data_2)
                
                # Verify that the LLM received the injected memory
                # The LLM node calls chat_completion(messages=...)
                call_kwargs = bridge_instance.chat_completion.call_args.kwargs
                messages_sent = call_kwargs['messages']
                
                # Expecting: [System Message (Memory), User Message]
                assert len(messages_sent) >= 2
                assert messages_sent[0]['role'] == 'system'
                assert "My secret code is 1234" in messages_sent[0]['content']