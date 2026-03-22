"""
Tests for GoalPursuitExecutor — integrated planning, step evaluation, replanning.
"""
import json
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from modules.agent_loop.node import GoalPursuitExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_executor():
    """Create a GoalPursuitExecutor with patched LLMBridge and sandbox."""
    with patch("modules.agent_loop.node.LLMBridge"):
        ex = GoalPursuitExecutor()
    ex.llm = MagicMock()
    ex._sandbox = MagicMock()
    return ex


def llm_response(content: str) -> dict:
    """Build a minimal valid LLM response dict."""
    return {"choices": [{"message": {"content": content, "tool_calls": []}}]}


def plan_json(*actions) -> str:
    """Serialize a list of step dicts to JSON."""
    return json.dumps([
        {"step": i + 1, "action": a, "target": f"target_{i+1}", "goal": f"goal_{i+1}"}
        for i, a in enumerate(actions)
    ])


def eval_json(success: bool, reason: str = "ok", extracted: str = "result") -> str:
    return json.dumps({"success": success, "reason": reason, "extracted_result": extracted})


MESSAGES = [{"role": "user", "content": "Research AI trends and summarize them"}]


# ---------------------------------------------------------------------------
# _parse_plan_response
# ---------------------------------------------------------------------------

async def test_parse_plan_response_valid():
    ex = make_executor()
    raw = plan_json("Search", "Summarize")
    result = await ex._parse_plan_response(raw, max_steps=10)
    assert len(result) == 2
    assert result[0]["action"] == "Search"
    assert result[1]["action"] == "Summarize"
    assert result[0]["step"] == 1
    assert result[1]["step"] == 2


async def test_parse_plan_response_respects_max_steps():
    ex = make_executor()
    raw = plan_json("A", "B", "C", "D", "E")
    result = await ex._parse_plan_response(raw, max_steps=3)
    assert len(result) == 3


async def test_parse_plan_response_invalid_json_returns_empty():
    ex = make_executor()
    result = await ex._parse_plan_response("not json at all", max_steps=10)
    assert result == []


async def test_parse_plan_response_step_offset():
    ex = make_executor()
    raw = plan_json("Finish")
    result = await ex._parse_plan_response(raw, max_steps=10, step_offset=2)
    assert result[0]["step"] == 3  # offset=2 → step 3


# ---------------------------------------------------------------------------
# _create_plan
# ---------------------------------------------------------------------------

async def test_create_plan_returns_structured_steps():
    ex = make_executor()
    ex.llm.chat_completion = AsyncMock(return_value=llm_response(plan_json("Step A", "Step B")))
    plan = await ex._create_plan("Do something complex", config={})
    assert len(plan) == 2
    assert plan[0]["action"] == "Step A"


async def test_create_plan_llm_failure_returns_empty():
    ex = make_executor()
    ex.llm.chat_completion = AsyncMock(return_value=None)
    plan = await ex._create_plan("goal", config={})
    assert plan == []


async def test_create_plan_strips_js_comments():
    ex = make_executor()
    raw = '// comment\n[{"step": 1, "action": "Do it", "target": "x", "goal": "g"}]'
    ex.llm.chat_completion = AsyncMock(return_value=llm_response(raw))
    plan = await ex._create_plan("goal", config={})
    assert len(plan) == 1
    assert plan[0]["action"] == "Do it"


# ---------------------------------------------------------------------------
# _evaluate_step
# ---------------------------------------------------------------------------

async def test_evaluate_step_success():
    ex = make_executor()
    ex.llm.chat_completion = AsyncMock(
        return_value=llm_response(eval_json(True, "Step completed", "Found data"))
    )
    step = {"action": "Search", "target": "AI trends", "goal": "find AI trends"}
    result = await ex._evaluate_step(step, "Here are AI trends: ...", config={})
    assert result["success"] is True
    assert result["extracted_result"] == "Found data"


async def test_evaluate_step_failure():
    ex = make_executor()
    ex.llm.chat_completion = AsyncMock(
        return_value=llm_response(eval_json(False, "No relevant data found", ""))
    )
    step = {"action": "Search", "target": "AI trends", "goal": "find AI trends"}
    result = await ex._evaluate_step(step, "I could not find anything", config={})
    assert result["success"] is False
    assert "No relevant data" in result["reason"]


async def test_evaluate_step_llm_failure_returns_failure():
    ex = make_executor()
    ex.llm.chat_completion = AsyncMock(return_value=None)
    step = {"action": "Search", "target": "x", "goal": "g"}
    result = await ex._evaluate_step(step, "some content", config={})
    assert result["success"] is False
    assert "Evaluator LLM failed" in result["reason"]


async def test_evaluate_step_fallback_unparseable_returns_failure():
    """When JSON parse fails, returns fail-safe False (no keyword guessing)."""
    ex = make_executor()
    ex.llm.chat_completion = AsyncMock(return_value=llm_response("Yes, this was completed successfully"))
    step = {"action": "Search", "target": "x", "goal": "g"}
    result = await ex._evaluate_step(step, "content", config={})
    assert result["success"] is False
    assert "could not be parsed" in result["reason"]


# ---------------------------------------------------------------------------
# _replan
# ---------------------------------------------------------------------------

async def test_replan_returns_revised_steps():
    ex = make_executor()
    new_plan = json.dumps([
        {"step": 2, "action": "Alternative search", "target": "AI news", "goal": "find news"}
    ])
    ex.llm.chat_completion = AsyncMock(return_value=llm_response(new_plan))

    completed = [{"step": 1, "action": "Setup", "target": "env"}]
    results = [{"extracted_result": "env ready"}]
    failed = {"step": 2, "action": "Search", "target": "trends"}

    new_steps = await ex._replan(
        original_request="Research AI",
        completed_steps=completed,
        step_results=results,
        failed_step=failed,
        failure_reason="Tool not found",
        config={},
    )
    assert len(new_steps) == 1
    assert new_steps[0]["action"] == "Alternative search"
    assert new_steps[0]["step"] == 2  # step_offset = len(completed) = 1, so step = 1+1 = 2


async def test_replan_llm_failure_returns_empty():
    ex = make_executor()
    ex.llm.chat_completion = AsyncMock(return_value=None)
    result = await ex._replan("goal", [], [], {}, "reason", config={})
    assert result == []


# ---------------------------------------------------------------------------
# _build_plan_context
# ---------------------------------------------------------------------------

def test_build_plan_context_shows_progress():
    ex = make_executor()
    completed = [{"step": 1, "action": "Search", "target": "web"}]
    remaining = [
        {"step": 2, "action": "Summarize", "target": "results"},
        {"step": 3, "action": "Write", "target": "report"},
    ]
    results = [{"extracted_result": "found 10 items"}]
    ctx = ex._build_plan_context(completed, remaining, current_idx=0, step_results=results)
    assert "1/3" in ctx
    assert "DONE" in ctx
    assert "CURRENT" in ctx
    assert "found 10 items" in ctx


# ---------------------------------------------------------------------------
# receive() — no messages
# ---------------------------------------------------------------------------

async def test_receive_no_messages_returns_error():
    ex = make_executor()
    result = await ex.receive({}, config={})
    assert result.get("agent_loop_error") == "No messages provided"
    assert result["content"] == ""


async def test_receive_no_user_message_returns_error():
    ex = make_executor()
    result = await ex.receive(
        {"messages": [{"role": "system", "content": "system msg"}]},
        config={},
    )
    assert result.get("agent_loop_error") == "No user message found"


# ---------------------------------------------------------------------------
# receive() — happy path: single step succeeds
# ---------------------------------------------------------------------------

async def test_receive_single_step_success():
    ex = make_executor()

    plan_resp = llm_response(plan_json("Research AI"))
    step_resp = llm_response("AI is growing rapidly in 2025.")
    eval_resp = llm_response(eval_json(True, "Information found", "AI is growing rapidly in 2025."))

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive({"messages": MESSAGES}, config={"enable_synthesis": False})

    assert result["content"]
    assert len(result["completed_steps"]) == 1
    assert result["completed_steps"][0]["action"] == "Research AI"
    assert result["replan_count"] == 0
    assert result["remaining_steps"] == []


# ---------------------------------------------------------------------------
# receive() — multi-step with synthesis
# ---------------------------------------------------------------------------

async def test_receive_multi_step_with_synthesis():
    ex = make_executor()

    plan_resp = llm_response(plan_json("Gather data", "Analyze", "Summarize"))
    step1_resp = llm_response("Data gathered.")
    eval1_resp = llm_response(eval_json(True, "gathered", "Data gathered."))
    step2_resp = llm_response("Analysis done.")
    eval2_resp = llm_response(eval_json(True, "analyzed", "Analysis done."))
    step3_resp = llm_response("Summary written.")
    eval3_resp = llm_response(eval_json(True, "written", "Summary written."))
    synth_resp = llm_response("Final synthesis: Data gathered, analyzed, and summarized.")

    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp,
        step1_resp, eval1_resp,
        step2_resp, eval2_resp,
        step3_resp, eval3_resp,
        synth_resp,
    ])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": True},
        )

    assert len(result["completed_steps"]) == 3
    assert "synthesis" in result["content"].lower() or "summarized" in result["content"].lower()
    assert result["replan_count"] == 0


# ---------------------------------------------------------------------------
# receive() — step fails, triggers replan, then succeeds
# ---------------------------------------------------------------------------

async def test_receive_step_fails_replans_and_succeeds():
    ex = make_executor()

    plan_resp = llm_response(plan_json("Search web"))
    step1_fail_resp = llm_response("Could not search.")
    eval1_fail = llm_response(eval_json(False, "Search tool not available", ""))
    replan_resp = llm_response(json.dumps([
        {"step": 1, "action": "Use cached data", "target": "cache", "goal": "get data"}
    ]))
    step1_ok_resp = llm_response("Retrieved cached data.")
    eval1_ok = llm_response(eval_json(True, "Retrieved", "Cached data retrieved."))

    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp,
        step1_fail_resp, eval1_fail,
        replan_resp,
        step1_ok_resp, eval1_ok,
    ])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False, "max_replan_depth": 3},
        )

    assert len(result["completed_steps"]) == 1
    assert result["replan_count"] == 1
    assert result["completed_steps"][0]["action"] == "Use cached data"


# ---------------------------------------------------------------------------
# receive() — replan depth exceeded
# ---------------------------------------------------------------------------

async def test_receive_replan_depth_exceeded_stops():
    ex = make_executor()

    plan_resp = llm_response(plan_json("Impossible task"))
    step_fail = llm_response("")  # empty content → structural failure
    eval_fail = llm_response(eval_json(False, "No content produced", ""))
    replan_resp = llm_response(json.dumps([
        {"step": 1, "action": "Try again", "target": "x", "goal": "g"}
    ]))

    # Each replan attempt: fail → eval → replan
    # With max_replan_depth=2 and starting at 0:
    # attempt 0: fail → replan (count=1)
    # attempt 1: fail → replan (count=2)
    # attempt 2: count >= max → stop
    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp,
        step_fail, eval_fail, replan_resp,   # replan_count → 1
        step_fail, eval_fail, replan_resp,   # replan_count → 2
        step_fail, eval_fail,                # hits guard
    ])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False, "max_replan_depth": 2},
        )

    assert result["replan_count"] >= 2
    assert result["remaining_steps"] or result["replan_depth_exceeded"]


# ---------------------------------------------------------------------------
# receive() — step evaluation disabled
# ---------------------------------------------------------------------------

async def test_receive_evaluation_disabled_uses_structural_check():
    ex = make_executor()

    plan_resp = llm_response(plan_json("Do something"))
    step_resp = llm_response("Done!")

    # With enable_step_evaluation=False, only plan + step LLM calls are made
    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False, "enable_step_evaluation": False},
        )

    assert len(result["completed_steps"]) == 1
    # Evaluator LLM not called (only 2 calls total: plan + step)
    assert ex.llm.chat_completion.call_count == 2


# ---------------------------------------------------------------------------
# receive() — fallback to direct hybrid when plan is empty
# ---------------------------------------------------------------------------

async def test_receive_empty_plan_falls_back_to_hybrid():
    ex = make_executor()

    # First call: plan LLM returns empty array → fallback
    plan_resp = llm_response("[]")
    step_resp = llm_response("Direct answer without planning.")

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive({"messages": MESSAGES}, config={})

    assert result["content"] == "Direct answer without planning."
    assert result["plan"] == []
    assert result["completed_steps"] == []


# ---------------------------------------------------------------------------
# receive() — timeout
# ---------------------------------------------------------------------------

async def test_receive_timeout_returns_error():
    ex = make_executor()

    async def slow(*args, **kwargs):
        await asyncio.sleep(10)
        return llm_response("Done")

    ex.llm.chat_completion = AsyncMock(side_effect=slow)

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"timeout": 0.1},
        )

    assert "timed out" in result.get("agent_loop_error", "").lower()
    assert result["content"] == ""


# ---------------------------------------------------------------------------
# receive() — unexpected exception
# ---------------------------------------------------------------------------

async def test_receive_unexpected_exception_returns_error():
    """An exception that escapes the internal loop surfaces as agent_loop_error."""
    ex = make_executor()

    # Patch _create_plan so the exception bypasses _llm_with_retry's try/except
    with patch.object(ex, "_create_plan", side_effect=RuntimeError("boom")), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive({"messages": MESSAGES}, config={})

    assert "boom" in result.get("agent_loop_error", "")
    assert result["content"] == ""


# ---------------------------------------------------------------------------
# receive() — repl_state shared across steps
# ---------------------------------------------------------------------------

async def test_receive_repl_state_shared_across_steps():
    """Variables stored in step 1 should be visible to step 2."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Fetch data", "Process data"))
    step1_resp = llm_response("Fetched 100 items.")
    eval1 = llm_response(eval_json(True, "ok", "100 items"))
    step2_resp = llm_response("Processed 100 items successfully.")
    eval2 = llm_response(eval_json(True, "ok", "Processed"))

    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp, step1_resp, eval1, step2_resp, eval2,
    ])

    captured_repl_states = []

    original_run = ex._run_hybrid_loop

    async def capturing_run(*args, repl_state, **kwargs):
        captured_repl_states.append(repl_state)
        # Simulate storing a variable in step 1
        if not captured_repl_states[0].get("_step_1_done"):
            repl_state["variables"]["step1_output"] = "100 items"
            repl_state["_step_1_done"] = True
        result = await original_run(*args, repl_state=repl_state, **kwargs)
        return result

    with patch.object(ex, "_run_hybrid_loop", side_effect=capturing_run), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False},
        )

    # Both steps should have received the same repl_state object
    assert len(captured_repl_states) == 2
    assert captured_repl_states[0] is captured_repl_states[1]
    assert "step1_output" in captured_repl_states[1]["variables"]


# ---------------------------------------------------------------------------
# receive() — output keys present
# ---------------------------------------------------------------------------

async def test_receive_output_keys_present():
    ex = make_executor()

    plan_resp = llm_response(plan_json("Do it"))
    step_resp = llm_response("Done.")
    eval_resp = llm_response(eval_json(True, "ok", "Done."))

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False},
        )

    required_keys = {
        "content", "messages", "plan", "completed_steps", "step_results",
        "remaining_steps", "replan_count", "goal_pursuit_trace",
        "agent_loop_trace", "iterations",
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing output keys: {missing}"


# ---------------------------------------------------------------------------
# receive() — preserves existing input fields
# ---------------------------------------------------------------------------

async def test_receive_preserves_input_fields():
    ex = make_executor()

    plan_resp = llm_response(plan_json("Task"))
    step_resp = llm_response("Done.")
    eval_resp = llm_response(eval_json(True, "ok", "Done."))

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    input_data = {
        "messages": MESSAGES,
        "session_id": "sess-123",
        "_memory_context": "User likes concise answers.",
        "custom_field": 42,
    }

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(input_data, config={"enable_synthesis": False})

    assert result["session_id"] == "sess-123"
    assert result["custom_field"] == 42


# ---------------------------------------------------------------------------
# get_executor_class routing
# ---------------------------------------------------------------------------

async def test_get_executor_class_goal_pursuit():
    from modules.agent_loop.node import get_executor_class
    cls = await get_executor_class("goal_pursuit")
    assert cls is GoalPursuitExecutor


async def test_get_executor_class_unknown():
    from modules.agent_loop.node import get_executor_class
    cls = await get_executor_class("not_a_real_node")
    assert cls is None


# ---------------------------------------------------------------------------
# _evaluate_step — markdown fence stripping
# ---------------------------------------------------------------------------

async def test_evaluate_step_strips_markdown_fences():
    """JSON wrapped in ```json ... ``` code fences is correctly parsed."""
    ex = make_executor()
    fenced = '```json\n{"success": true, "reason": "all good", "extracted_result": "value"}\n```'
    ex.llm.chat_completion = AsyncMock(return_value=llm_response(fenced))
    step = {"action": "Do", "target": "x", "goal": "g"}
    result = await ex._evaluate_step(step, "content", config={})
    assert result["success"] is True
    assert result["reason"] == "all good"
    assert result["extracted_result"] == "value"


# ---------------------------------------------------------------------------
# receive() — consecutive failures skip step after max_step_retries
# ---------------------------------------------------------------------------

async def test_receive_step_skipped_after_max_retries():
    """Step that fails max_step_retries consecutive times is skipped, not infinitely replanned."""
    ex = make_executor()

    # Plan has 2 steps; step 1 fails twice (max_step_retries=2) and gets skipped.
    plan_resp = llm_response(plan_json("Failing step", "Working step"))
    step1_fail = llm_response("Error.")
    eval1_fail = llm_response(eval_json(False, "failed", ""))
    step2_resp = llm_response("Done.")
    eval2_ok = llm_response(eval_json(True, "ok", "Done."))

    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp,
        step1_fail, eval1_fail,   # consecutive_failures = 1 → replan
        llm_response(json.dumps([  # replan: same step 1
            {"step": 1, "action": "Failing step", "target": "x", "goal": "g"},
            {"step": 2, "action": "Working step", "target": "y", "goal": "g2"},
        ])),
        step1_fail, eval1_fail,   # consecutive_failures = 1 on new plan → replan again? no — we track from 0
        # Actually: after replan, consecutive_failures=0. So 2nd fail → consecutive_failures=1 again.
        # We need TWO more fails on the same step without replan in between to hit max_step_retries=2.
    ])

    # Simpler: use max_step_retries=1 so a single failure triggers a skip
    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp,
        step1_fail, eval1_fail,   # consecutive_failures=1 >= max_step_retries=1 → SKIP step 1
        step2_resp, eval2_ok,     # step 2 succeeds
    ])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False, "max_step_retries": 1},
        )

    # Step 1 was skipped, step 2 completed
    assert len(result["completed_steps"]) == 1
    assert result["completed_steps"][0]["action"] == "Working step"
    # No replan was consumed — skip doesn't burn replan budget
    assert result["replan_count"] == 0
    # Step 1 trace should show it was skipped
    skipped = [t for t in result["goal_pursuit_trace"] if t.get("skipped")]
    assert len(skipped) == 1


# ---------------------------------------------------------------------------
# receive() — step result stored in repl_state variables on success
# ---------------------------------------------------------------------------

async def test_receive_step_result_stored_in_repl_state():
    """Successful step stores its full content in repl_state['variables']['step_N_result']."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Gather info"))
    step_resp = llm_response("Found 42 results.")
    eval_resp = llm_response(eval_json(True, "ok", "Found 42 results."))

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    captured_repl_state = {}

    original_run = ex._run_hybrid_loop

    async def capture_repl(*args, repl_state, **kwargs):
        result = await original_run(*args, repl_state=repl_state, **kwargs)
        captured_repl_state.update(repl_state)
        return result

    with patch.object(ex, "_run_hybrid_loop", side_effect=capture_repl), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False},
        )

    # The step_1_result variable should be stored after success
    assert "step_1_result" in captured_repl_state.get("variables", {})
    # And it should contain the step's LLM output
    assert "42" in captured_repl_state["variables"]["step_1_result"]


# ---------------------------------------------------------------------------
# receive() — per-step timeout treats step as failure
# ---------------------------------------------------------------------------

async def test_receive_step_timeout_treated_as_failure():
    """When step_timeout fires, the step is treated as failed (skipped via max_step_retries=1)."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Slow step"))
    eval_fail = llm_response(eval_json(False, "timed out", ""))

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp])

    async def slow_hybrid(*args, **kwargs):
        await asyncio.sleep(10)  # will be cancelled by wait_for
        return None, 0, False

    with patch.object(ex, "_run_hybrid_loop", side_effect=slow_hybrid), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={
                "enable_synthesis": False,
                "step_timeout": 0.05,   # 50ms — fires immediately
                "max_step_retries": 1,  # skip after 1 failure
            },
        )

    # Step timed out and was skipped; no completed steps
    assert len(result["completed_steps"]) == 0
    timed_out_traces = [t for t in result["goal_pursuit_trace"] if t.get("timed_out")]
    assert len(timed_out_traces) == 1
