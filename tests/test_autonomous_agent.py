"""Tests for the experimental autonomous agent runtime."""

import json

import pytest

from modules.autonomous_agent.core.communicator import AgentCommunicator
from modules.autonomous_agent.core.sandbox import FlowValidationError
from modules.autonomous_agent.core.service import AgentService
from modules.autonomous_agent.core.types import (
    AgentConfig,
    Experience,
    FlowSpec,
    Perception,
)
from modules.autonomous_agent.nodes import NODE_EXECUTORS


def make_agent(tmp_path, **overrides) -> AgentService:
    """Build an agent whose runtime files live under ``tmp_path``."""
    config = AgentConfig(
        agent_id=overrides.pop("agent_id", "test-agent"),
        persistence={
            "enabled": True,
            "state_path": str(tmp_path / "state.json"),
        },
        **overrides,
    )
    agent = AgentService(config)
    agent.flows_store_path = tmp_path / "ai_flows.json"
    return agent


@pytest.mark.asyncio
async def test_handle_user_message_records_perception_and_chat_ack(tmp_path):
    agent = make_agent(tmp_path)
    assert await agent.initialize() is True

    perception = await agent.handle_user_message("chat", "hello", "user-1")

    assert perception.content == "hello"
    assert perception.metadata["user_id"] == "user-1"
    assert agent.state.recent_perceptions[-1]["content"] == "hello"
    messages = agent.communicator.get_pending_chat_messages()
    assert messages
    assert messages[-1]["content"] == "Received. Processing..."


@pytest.mark.asyncio
async def test_memory_persists_across_service_instances(tmp_path):
    first = make_agent(tmp_path)
    assert await first.initialize() is True

    await first.memory.episodic.store_experience(
        Experience(
            type="user_interaction",
            perception=Perception(source="user", content="persist me"),
            result={"ok": True},
        )
    )
    await first.memory.semantic.store_fact(
        "NeuroCore uses flows", "architecture"
    )
    await first.memory.procedural.store_pattern(
        "deploy", {"steps": ["test"]}, occurrences=4
    )
    await first._save_state()
    await first.stop()

    second = make_agent(tmp_path)
    assert await second.initialize() is True

    stats = second.memory.get_all_stats()
    assert stats["episodic"]["count"] == 1
    assert stats["semantic"]["count"] == 1
    assert stats["procedural"]["count"] == 1
    assert await second.memory.semantic.search("flows")
    await second.stop()


@pytest.mark.asyncio
async def test_corrupt_state_is_quarantined(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{not valid json", encoding="utf-8")

    agent = make_agent(tmp_path)
    assert await agent.initialize() is True

    assert any(error["phase"] == "memory_loading" for error in agent.state.errors)
    assert not state_path.exists()
    assert list(tmp_path.glob("state.json.corrupt-*"))


@pytest.mark.asyncio
async def test_valid_json_state_with_wrong_shape_is_quarantined(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("[]", encoding="utf-8")

    agent = make_agent(tmp_path)
    assert await agent.initialize() is True

    assert any(error["phase"] == "memory_loading" for error in agent.state.errors)
    assert not state_path.exists()
    assert list(tmp_path.glob("state.json.corrupt-*"))


@pytest.mark.asyncio
async def test_persistence_can_be_disabled(tmp_path):
    agent = AgentService(
        AgentConfig(
            agent_id="no-persist",
            persistence={"enabled": False},
        )
    )
    agent.flows_store_path = tmp_path / "ai_flows.json"

    assert await agent.initialize() is True
    await agent._save_state()

    assert agent._state_path() is None


@pytest.mark.asyncio
async def test_flow_creation_registers_ownership(tmp_path):
    agent = make_agent(tmp_path)
    await agent.initialize()

    flow = await agent.create_flow(
        FlowSpec(
            description="chat flow",
            required_nodes=["chat_input", "llm_module", "chat_output"],
        )
    )

    assert flow["metadata"]["creator"] == "test-agent"
    assert flow["id"] in agent.sandbox.get_owned_flows()
    assert flow["nodes"]


@pytest.mark.asyncio
async def test_sandbox_rejects_flow_over_configured_limit(tmp_path):
    agent = make_agent(
        tmp_path,
        sandbox={"max_flow_nodes": 6, "max_flow_complexity": 100},
    )
    await agent.initialize()

    with pytest.raises(FlowValidationError):
        await agent.create_flow(
            FlowSpec(description="too large", required_nodes=["a", "b", "c"])
        )


@pytest.mark.asyncio
async def test_load_existing_flows_adopts_only_owned(tmp_path):
    flows = {
        "owned-flow": {"id": "owned-flow", "metadata": {"creator": "test-agent"}},
        "other-flow": {"id": "other-flow", "metadata": {"creator": "someone-else"}},
    }
    (tmp_path / "ai_flows.json").write_text(
        json.dumps(flows), encoding="utf-8"
    )

    agent = make_agent(tmp_path)
    await agent.initialize()

    assert "owned-flow" in agent.flow_creator.created_flows
    assert "owned-flow" in agent.sandbox.get_owned_flows()
    assert "other-flow" not in agent.flow_creator.created_flows


@pytest.mark.asyncio
async def test_load_existing_flows_skips_malformed_metadata(tmp_path):
    flows = {
        "string-metadata": {"id": "string-metadata", "metadata": "not-a-dict"},
        "null-metadata": {"id": "null-metadata", "metadata": None},
        "missing-metadata": {"id": "missing-metadata"},
        "owned-flow": {"id": "owned-flow", "metadata": {"creator": "test-agent"}},
    }
    (tmp_path / "ai_flows.json").write_text(
        json.dumps(flows), encoding="utf-8"
    )

    agent = make_agent(tmp_path)
    assert await agent.initialize() is True

    assert agent.state.errors == []
    assert "owned-flow" in agent.flow_creator.created_flows
    assert "string-metadata" not in agent.flow_creator.created_flows
    assert "null-metadata" not in agent.flow_creator.created_flows
    assert "missing-metadata" not in agent.flow_creator.created_flows


@pytest.mark.asyncio
async def test_communicator_buffers_and_drains_chat_messages():
    communicator = AgentCommunicator(
        "comm-test", AgentConfig(agent_id="comm-test")
    )

    await communicator.notify_user("heads up")

    messages = communicator.get_pending_chat_messages()
    assert messages[0]["content"] == "heads up"
    assert "timestamp" in messages[0]
    assert communicator.get_pending_chat_messages() == []


@pytest.mark.asyncio
async def test_agent_node_processes_single_message(tmp_path):
    config_path = tmp_path / "agent.json"
    config_path.write_text(
        json.dumps(
            {
                "agent_id": "node-test",
                "name": "Node Test",
                "persistence": {
                    "enabled": True,
                    "state_path": str(tmp_path / "node-state.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    node = NODE_EXECUTORS["autonomous_agent_node"]()

    result = await node.receive(
        {"message": "hello node", "channel": "chat", "user_id": "u1"},
        {
            "agent_config_path": str(config_path),
            "auto_start": False,
            "run_mode": "single",
        },
    )

    assert result["status"] == "message_queued"
    assert result["perception"]["content"] == "hello node"
    await node.shutdown()
