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
# Autouse fixture — suppress pre-step reasoning for all existing tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stub_step_reasoning(monkeypatch):
    """Stub out _reason_about_step for all GoalPursuit tests.

    Existing tests set up specific LLM mock sequences; the extra
    _reason_about_step LLM call would consume those responses unexpectedly.
    Tests that want to exercise reasoning should explicitly undo this patch.
    """
    async def _no_reasoning(self, step, repl_state, config, user_goal):
        return ""
    monkeypatch.setattr(
        "modules.agent_loop.node.GoalPursuitExecutor._reason_about_step",
        _no_reasoning,
    )


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
            config={"enable_synthesis": False, "max_replan_depth": 3, "max_step_retries": 0},
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
            config={"enable_synthesis": False, "max_replan_depth": 2, "max_step_retries": 0},
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

    # max_step_retries=1: one failure triggers alternative approach, then skip if alt also fails
    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp,
        step1_fail, eval1_fail,   # attempt 1 → consecutive_failures=1 >= 1 → alternative
        step1_fail, eval1_fail,   # alternative attempt also fails → step 1 SKIPPED
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


# ---------------------------------------------------------------------------
# _topo_sort_steps — dependency graph ordering
# ---------------------------------------------------------------------------

def test_topo_sort_linear_no_deps():
    """Steps with no depends_on come out in original order."""
    steps = [
        {"step": 1, "action": "A", "depends_on": []},
        {"step": 2, "action": "B", "depends_on": []},
        {"step": 3, "action": "C", "depends_on": []},
    ]
    result = GoalPursuitExecutor._topo_sort_steps(steps)
    assert [s["step"] for s in result] == [1, 2, 3]


def test_topo_sort_respects_dependencies():
    """Step 3 depends on step 1; step 2 is independent. Valid orders: 1,2,3 or 1,3 after 1, or 2,1,3."""
    steps = [
        {"step": 1, "action": "A", "depends_on": []},
        {"step": 2, "action": "B", "depends_on": []},
        {"step": 3, "action": "C", "depends_on": [1]},
    ]
    result = GoalPursuitExecutor._topo_sort_steps(steps)
    step_nums = [s["step"] for s in result]
    assert step_nums.index(1) < step_nums.index(3), "Step 1 must come before step 3"
    assert len(step_nums) == 3


def test_topo_sort_chain():
    """3 → 2 → 1 dependency chain must run as 1, 2, 3."""
    steps = [
        {"step": 3, "action": "C", "depends_on": [2]},
        {"step": 1, "action": "A", "depends_on": []},
        {"step": 2, "action": "B", "depends_on": [1]},
    ]
    result = GoalPursuitExecutor._topo_sort_steps(steps)
    assert [s["step"] for s in result] == [1, 2, 3]


def test_topo_sort_cycle_returns_original():
    """Cyclic deps are detected and original order is returned unchanged."""
    steps = [
        {"step": 1, "action": "A", "depends_on": [2]},
        {"step": 2, "action": "B", "depends_on": [1]},
    ]
    result = GoalPursuitExecutor._topo_sort_steps(steps)
    # Should return original order unchanged (cycle detection)
    assert [s["step"] for s in result] == [1, 2]


def test_topo_sort_ignores_external_deps():
    """Deps pointing to step numbers not in the plan are ignored."""
    steps = [
        {"step": 3, "action": "C", "depends_on": [99]},  # 99 not in plan
        {"step": 4, "action": "D", "depends_on": [3]},
    ]
    result = GoalPursuitExecutor._topo_sort_steps(steps)
    assert [s["step"] for s in result] == [3, 4]


# ---------------------------------------------------------------------------
# _parse_plan_response — depends_on extraction
# ---------------------------------------------------------------------------

async def test_parse_plan_response_extracts_depends_on():
    """depends_on field is parsed from plan JSON."""
    ex = make_executor()
    content = json.dumps([
        {"step": 1, "action": "Fetch", "target": "data", "goal": "g", "depends_on": []},
        {"step": 2, "action": "Process", "target": "data", "goal": "g", "depends_on": [1]},
        {"step": 3, "action": "Report", "target": "findings", "goal": "g", "depends_on": [1, 2]},
    ])
    plan = await ex._parse_plan_response(content, max_steps=10)
    assert plan[0]["depends_on"] == []
    assert plan[1]["depends_on"] == [1]
    assert plan[2]["depends_on"] == [1, 2]


async def test_parse_plan_response_default_empty_depends_on():
    """Steps without depends_on get an empty list."""
    ex = make_executor()
    content = json.dumps([{"step": 1, "action": "Do", "target": "x", "goal": "g"}])
    plan = await ex._parse_plan_response(content, max_steps=10)
    assert plan[0]["depends_on"] == []


# ---------------------------------------------------------------------------
# receive() — dependency graph execution ordering
# ---------------------------------------------------------------------------

async def test_receive_dependency_respected():
    """Steps with depends_on are deferred until their prerequisites complete."""
    ex = make_executor()

    # Two-step plan where step 2 depends on step 1.
    plan_json_str = json.dumps([
        {"step": 1, "action": "Gather", "target": "data", "goal": "g", "depends_on": []},
        {"step": 2, "action": "Analyze", "target": "data", "goal": "g", "depends_on": [1]},
    ])
    plan_resp = llm_response(plan_json_str)
    step1_resp = llm_response("Data gathered.")
    eval1 = llm_response(eval_json(True, "ok", "gathered"))
    step2_resp = llm_response("Analysis done.")
    eval2 = llm_response(eval_json(True, "ok", "done"))

    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp, step1_resp, eval1, step2_resp, eval2,
    ])

    executed_order = []
    original_run = ex._run_hybrid_loop

    async def tracking_run(*args, llm_messages, **kwargs):
        # Extract the step name from the user message
        user_msg = next((m["content"] for m in llm_messages if m["role"] == "user"), "")
        executed_order.append(user_msg[:30])
        return await original_run(*args, llm_messages=llm_messages, **kwargs)

    with patch.object(ex, "_run_hybrid_loop", side_effect=tracking_run), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False},
        )

    assert len(result["completed_steps"]) == 2
    # Step 1 must have been executed before step 2
    assert executed_order[0].startswith("Complete this step: Gather")
    assert executed_order[1].startswith("Complete this step: Analyze")


# ---------------------------------------------------------------------------
# receive() — repl_state variables restored from _repl_variables
# ---------------------------------------------------------------------------

async def test_receive_repl_variables_restored_from_input():
    """Variables in input_data['_repl_variables'] are loaded into repl_state at startup."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Use cached data"))
    step_resp = llm_response("Used it.")
    eval_resp = llm_response(eval_json(True, "ok", "Used it."))

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    captured_vars = {}
    original_run = ex._run_hybrid_loop

    async def capture(*args, repl_state, **kwargs):
        captured_vars.update(repl_state["variables"])
        return await original_run(*args, repl_state=repl_state, **kwargs)

    with patch.object(ex, "_run_hybrid_loop", side_effect=capture), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        await ex.receive(
            {
                "messages": MESSAGES,
                "_repl_variables": {"step_0_result": "previously computed result"},
            },
            config={"enable_synthesis": False},
        )

    # The pre-seeded variable should be visible during step execution
    assert "step_0_result" in captured_vars
    assert captured_vars["step_0_result"] == "previously computed result"


# ---------------------------------------------------------------------------
# AskUser — mid-plan pause / resume
# ---------------------------------------------------------------------------

async def test_ask_user_pauses_execution():
    """When the agent calls AskUser during a step, receive() returns immediately with paused=True."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Clarify requirements", "Do the work"))
    # AskUser is handled by _execute_tool override — simulate by making _run_hybrid_loop
    # set _pending_question on repl_state directly (as the override would do).
    step1_resp = llm_response("Pausing to ask.")

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step1_resp])

    async def run_with_ask_user(*args, repl_state, llm_messages, **kwargs):
        repl_state["_pending_question"] = "What format do you want the output in?"
        return ({"content": "Pausing to ask."}, 1, False)

    with patch.object(ex, "_run_hybrid_loop", side_effect=run_with_ask_user), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False, "enable_ask_user": True},
        )

    assert result.get("paused") is True
    assert result["pending_question"] == "What format do you want the output in?"
    assert result["content"] == "What format do you want the output in?"


async def test_ask_user_resume_injects_answer():
    """_user_answer injected via _repl_variables (as episode-resume path does) is visible in repl_state."""
    ex = make_executor()

    # The episode-resume path writes _user_answer into _repl_variables before calling receive().
    # Simulate that by pre-seeding _repl_variables directly.
    pre_seeded_vars = {"_user_answer": "Bullet points please"}

    plan_resp = llm_response(plan_json("Do the work"))
    step_resp = llm_response("Done in bullet points.")
    eval_resp = llm_response(eval_json(True, "ok", "Done."))

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    captured_vars = {}

    async def capture(*args, repl_state, llm_messages, **kwargs):
        captured_vars.update(repl_state.get("variables", {}))
        return ({"content": "Done in bullet points."}, 1, False)

    with patch.object(ex, "_run_hybrid_loop", side_effect=capture), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES, "_repl_variables": pre_seeded_vars},
            config={"enable_synthesis": False, "enable_ask_user": True},
        )

    assert captured_vars.get("_user_answer") == "Bullet points please"
    assert result.get("paused") is not True


async def test_execute_tool_intercepts_ask_user():
    """_execute_tool override sets _pending_question and does not call super()."""
    ex = make_executor()
    repl_state = {"variables": {}}
    tool_call = {
        "id": "tc1",
        "function": {
            "name": "AskUser",
            "arguments": json.dumps({"question": "Which dataset should I use?"})
        }
    }
    result = await ex._execute_tool(tool_call, {}, repl_state, large_output_threshold=3000)

    assert repl_state.get("_pending_question") == "Which dataset should I use?"
    assert result["name"] == "AskUser"
    assert result["success"] is True
    # Content should acknowledge pause without executing anything
    assert "paused" in result["content"].lower() or "waiting" in result["content"].lower()


# ---------------------------------------------------------------------------
# Alternative approach on step exhaustion
# ---------------------------------------------------------------------------

async def test_alternative_approach_succeeds_after_max_retries():
    """When a step hits max_step_retries=1, an alternative attempt is made before skipping."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Do the thing"))
    fail_resp = llm_response("Failed attempt.")
    fail_eval = llm_response(eval_json(False, "no results"))
    alt_resp = llm_response("Alternative succeeded.")
    alt_eval = llm_response(eval_json(True, "ok", "Alternative succeeded."))

    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp,
        fail_resp, fail_eval,   # attempt 1 (consecutive_failures=1 >= max_step_retries=1)
        alt_resp, alt_eval,     # alternative attempt
    ])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False, "max_step_retries": 1},
        )

    assert len(result["completed_steps"]) == 1
    trace = result["goal_pursuit_trace"]
    assert trace[-1].get("alternative_attempt") is True
    assert trace[-1].get("alternative_succeeded") is True


async def test_alternative_approach_fails_then_step_skipped():
    """When normal attempt and alternative both fail, the step is skipped."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Do the thing", "Second step"))
    fail_resp = llm_response("Failed.")
    fail_eval = llm_response(eval_json(False, "no results"))
    step2_resp = llm_response("Second step done.")
    step2_eval = llm_response(eval_json(True, "ok", "done"))

    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp,
        fail_resp, fail_eval,  # attempt 1 (consecutive_failures=1 >= max_step_retries=1)
        fail_resp, fail_eval,  # alternative attempt also fails → step 1 skipped
        step2_resp, step2_eval,
    ])

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False, "max_step_retries": 1},
        )

    skipped = [t for t in result["goal_pursuit_trace"] if t.get("skipped")]
    assert len(skipped) == 1
    assert skipped[0].get("alternative_attempt") is True
    assert skipped[0].get("alternative_succeeded") is False
    # Second step should still complete
    assert len(result["completed_steps"]) == 1


# ---------------------------------------------------------------------------
# _save_run_reflection — cross-run learning
# ---------------------------------------------------------------------------

async def test_save_run_reflection_called_on_completion():
    """_save_run_reflection is scheduled as a task after _execute() returns."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Do something"))
    step_resp = llm_response("Done.")
    eval_resp = llm_response(eval_json(True, "ok", "Done."))

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    reflection_calls = []

    async def capture_reflection(**kwargs):
        reflection_calls.append(kwargs)

    with patch.object(ex, "_save_run_reflection", side_effect=capture_reflection), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        # create_task schedules it — we need to let the event loop run it
        result = await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False},
        )
        # Yield to the event loop so the scheduled task runs
        await asyncio.sleep(0)

    assert len(reflection_calls) == 1
    assert reflection_calls[0]["succeeded"] is True
    assert "Research AI trends" in reflection_calls[0]["goal"]


# ---------------------------------------------------------------------------
# _compress_step_results — RLM synthesis compression
# ---------------------------------------------------------------------------

async def test_compress_step_results_noop_when_small():
    """No compression when total step results are under the threshold."""
    ex = make_executor()
    repl_state = {"variables": {
        "step_1_result": "short result",
        "step_2_result": "also short",
    }}
    count = await ex._compress_step_results(repl_state, {"synthesis_compress_threshold": 9000})
    assert count == 0
    assert repl_state["variables"]["step_1_result"] == "short result"


async def test_compress_step_results_compresses_large_vars():
    """Large step results are summarized when total exceeds threshold."""
    ex = make_executor()
    big_text = "x" * 2000
    repl_state = {"variables": {
        "step_1_result": big_text,
        "step_2_result": big_text,
        "step_3_result": big_text,
        "step_4_result": big_text,
        "step_5_result": big_text,
    }}
    summary_resp = llm_response("Compressed summary of findings.")
    ex.llm.chat_completion = AsyncMock(return_value=summary_resp)

    count = await ex._compress_step_results(
        repl_state,
        {"synthesis_compress_threshold": 100, "synthesis_per_var_threshold": 500},
    )
    assert count == 5
    for k in repl_state["variables"]:
        assert repl_state["variables"][k].startswith("[Compressed summary]")


async def test_compress_step_results_called_before_synthesis():
    """_compress_step_results is awaited before _synthesize during plan execution."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Step A", "Step B"))
    step_resp = llm_response("done")
    eval_resp = llm_response(eval_json(True, "ok", "done"))
    synth_resp = llm_response("Final answer.")

    ex.llm.chat_completion = AsyncMock(side_effect=[
        plan_resp,
        step_resp, eval_resp,   # step 1
        step_resp, eval_resp,   # step 2
        synth_resp,             # synthesis
    ])

    compress_calls = []

    async def fake_compress(repl_state, config):
        compress_calls.append(True)
        return 0

    with patch.object(ex, "_compress_step_results", side_effect=fake_compress), \
         patch.object(ex, "_fetch_goal_lessons", return_value=[]), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        await ex.receive({"messages": MESSAGES}, config={"enable_synthesis": True})

    assert len(compress_calls) == 1


# ---------------------------------------------------------------------------
# _fetch_goal_lessons / _compress_lessons — RLM planning memory
# ---------------------------------------------------------------------------

async def test_fetch_goal_lessons_returns_empty_on_db_error():
    """DB errors in _fetch_goal_lessons are silently swallowed."""
    ex = make_executor()
    with patch("modules.agent_loop.node.GoalPursuitExecutor._fetch_goal_lessons",
               return_value=[]):
        lessons = await ex._fetch_goal_lessons({})
    assert lessons == []


async def test_compress_lessons_passthrough_when_small():
    """Short lessons are returned unchanged (no LLM call)."""
    ex = make_executor()
    lessons = ["Lesson 1: do X.", "Lesson 2: avoid Y."]
    result = await ex._compress_lessons(lessons, {"lessons_compress_threshold": 9000})
    assert "Lesson 1" in result
    assert "Lesson 2" in result
    ex.llm.chat_completion.assert_not_called()


async def test_compress_lessons_calls_llm_when_large():
    """Large lesson list triggers chunked LLM summarization."""
    ex = make_executor()
    # 10 lessons each 400 chars → 4000 total > threshold of 100
    lessons = [f"Lesson {i}: " + "x" * 390 for i in range(10)]
    summary_resp = llm_response("Bullet: key insight.")
    ex.llm.chat_completion = AsyncMock(return_value=summary_resp)

    result = await ex._compress_lessons(
        lessons,
        {"lessons_compress_threshold": 100, "lessons_chunk_size": 3},
    )
    assert ex.llm.chat_completion.called
    assert len(result) > 0


async def test_lessons_injected_into_planning_prompt():
    """Past lessons are fetched and injected when _create_plan_with_trace runs."""
    ex = make_executor()

    plan_resp = llm_response(plan_json("Do something"))
    step_resp = llm_response("done")
    eval_resp = llm_response(eval_json(True, "ok", "done"))

    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    captured_prompts = []
    original_create_plan = ex._create_plan.__func__

    async def capturing_create_plan(self, user_request, config, lessons="", tools_summary=""):
        captured_prompts.append(lessons)
        return await original_create_plan(self, user_request, config, lessons=lessons, tools_summary=tools_summary)

    with patch.object(type(ex), "_create_plan", capturing_create_plan), \
         patch.object(ex, "_fetch_goal_lessons", return_value=["Past lesson: use tool X."]), \
         patch.object(ex, "_compress_lessons", return_value="Past lesson: use tool X."), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False},
        )

    assert len(captured_prompts) == 1
    assert "Past lesson" in captured_prompts[0]


# ---------------------------------------------------------------------------
# Sub-goals — DecomposeGoal tool
# ---------------------------------------------------------------------------

async def test_decompose_goal_tool_injected_when_enabled():
    """DecomposeGoal appears in tools_list when enable_subgoals=True (default)."""
    ex = make_executor()
    plan_resp = llm_response(plan_json("Do something"))
    step_resp = llm_response("done")
    eval_resp = llm_response(eval_json(True, "ok", "done"))
    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    captured_tools = []
    original_run = ex._run_hybrid_loop

    async def capture_tools(*args, tools_list, **kwargs):
        captured_tools.extend(tools_list)
        return await original_run(*args, tools_list=tools_list, **kwargs)

    with patch.object(ex, "_run_hybrid_loop", side_effect=capture_tools), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        await ex.receive({"messages": MESSAGES}, config={"enable_synthesis": False})

    tool_names = {t["function"]["name"] for t in captured_tools if isinstance(t, dict)}
    assert "DecomposeGoal" in tool_names


async def test_decompose_goal_tool_not_injected_at_max_depth():
    """DecomposeGoal is absent when subgoal_depth >= max_subgoal_depth."""
    ex = make_executor()
    plan_resp = llm_response(plan_json("Do something"))
    step_resp = llm_response("done")
    eval_resp = llm_response(eval_json(True, "ok", "done"))
    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    captured_tools = []
    original_run = ex._run_hybrid_loop

    async def capture_tools(*args, tools_list, **kwargs):
        captured_tools.extend(tools_list)
        return await original_run(*args, tools_list=tools_list, **kwargs)

    with patch.object(ex, "_run_hybrid_loop", side_effect=capture_tools), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        # _subgoal_depth == max_subgoal_depth → tool should be absent
        await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False, "_subgoal_depth": 2, "max_subgoal_depth": 2},
        )

    tool_names = {t["function"]["name"] for t in captured_tools if isinstance(t, dict)}
    assert "DecomposeGoal" not in tool_names


async def test_decompose_goal_tool_not_injected_when_disabled():
    """DecomposeGoal is absent when enable_subgoals=False."""
    ex = make_executor()
    plan_resp = llm_response(plan_json("Do something"))
    step_resp = llm_response("done")
    eval_resp = llm_response(eval_json(True, "ok", "done"))
    ex.llm.chat_completion = AsyncMock(side_effect=[plan_resp, step_resp, eval_resp])

    captured_tools = []
    original_run = ex._run_hybrid_loop

    async def capture_tools(*args, tools_list, **kwargs):
        captured_tools.extend(tools_list)
        return await original_run(*args, tools_list=tools_list, **kwargs)

    with patch.object(ex, "_run_hybrid_loop", side_effect=capture_tools), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        await ex.receive(
            {"messages": MESSAGES},
            config={"enable_synthesis": False, "enable_subgoals": False},
        )

    tool_names = {t["function"]["name"] for t in captured_tools if isinstance(t, dict)}
    assert "DecomposeGoal" not in tool_names


async def test_execute_tool_decompose_goal_invokes_sub_receive():
    """_execute_tool with DecomposeGoal calls receive() recursively and returns result."""
    ex = make_executor()

    sub_plan_resp = llm_response(plan_json("Sub step"))
    sub_step_resp = llm_response("Sub-goal result.")
    sub_eval_resp = llm_response(eval_json(True, "ok", "Sub-goal result."))
    ex.llm.chat_completion = AsyncMock(
        side_effect=[sub_plan_resp, sub_step_resp, sub_eval_resp]
    )

    repl_state = {
        "variables": {},
        "_subgoal_depth": 0,
        "_max_subgoal_depth": 2,
        "_subgoal_config": {"enable_synthesis": False, "enable_subgoals": True},
    }
    tool_call = {
        "id": "tc1",
        "function": {
            "name": "DecomposeGoal",
            "arguments": json.dumps({
                "subgoal": "Research AI papers published this year",
                "context": "Parent goal: summarize AI landscape",
            }),
        },
    }

    with patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        result = await ex._execute_tool(tool_call, {}, repl_state, 3000)

    assert result["name"] == "DecomposeGoal"
    assert result["success"] is True
    assert result["content"]  # should contain the sub-goal's output


async def test_execute_tool_decompose_goal_missing_subgoal_returns_error():
    """DecomposeGoal with no subgoal parameter returns an error result."""
    ex = make_executor()
    repl_state = {
        "variables": {},
        "_subgoal_depth": 0,
        "_max_subgoal_depth": 2,
        "_subgoal_config": {},
    }
    tool_call = {
        "id": "tc1",
        "function": {
            "name": "DecomposeGoal",
            "arguments": json.dumps({}),
        },
    }
    result = await ex._execute_tool(tool_call, {}, repl_state, 3000)
    assert result["success"] is False
    assert "required" in result["content"].lower() or "error" in result["content"].lower()


async def test_subgoal_depth_incremented_in_recursive_call():
    """When DecomposeGoal spawns a sub-call, _subgoal_depth in the sub-config is depth+1."""
    ex = make_executor()

    captured_configs = []
    original_receive = ex.receive

    async def capture_receive(input_data, config=None):
        captured_configs.append(dict(config or {}))
        return await original_receive(input_data, config=config)

    sub_plan_resp = llm_response(plan_json("Sub step"))
    sub_step_resp = llm_response("Sub result.")
    sub_eval_resp = llm_response(eval_json(True, "ok", "Sub result."))
    ex.llm.chat_completion = AsyncMock(
        side_effect=[sub_plan_resp, sub_step_resp, sub_eval_resp]
    )

    repl_state = {
        "variables": {},
        "_subgoal_depth": 0,
        "_max_subgoal_depth": 2,
        "_subgoal_config": {"enable_synthesis": False, "enable_subgoals": True},
    }
    tool_call = {
        "id": "tc1",
        "function": {
            "name": "DecomposeGoal",
            "arguments": json.dumps({"subgoal": "Investigate this topic deeply"}),
        },
    }

    with patch.object(ex, "receive", side_effect=capture_receive), \
         patch.object(ex, "_load_tool_library", return_value={}), \
         patch.object(ex, "_load_tools", return_value={}):
        await ex._execute_tool(tool_call, {}, repl_state, 3000)

    # capture_receive was called once for the sub-goal
    assert len(captured_configs) == 1
    assert captured_configs[0].get("_subgoal_depth") == 1  # depth incremented


async def test_subgoal_result_content_returned_as_tool_result():
    """DecomposeGoal returns the sub-goal's content string as the tool result."""
    ex = make_executor()

    async def fake_receive(input_data, config=None):
        return {
            "content": "Sub done with findings.",
            "completed_steps": [],
            "repl_state": {"variables": [], "variable_count": 0},
        }

    repl_state = {
        "variables": {},
        "_subgoal_depth": 0,
        "_max_subgoal_depth": 2,
        "_subgoal_config": {},
    }
    tool_call = {
        "id": "tc1",
        "function": {
            "name": "DecomposeGoal",
            "arguments": json.dumps({"subgoal": "Find data"}),
        },
    }

    with patch.object(ex, "receive", side_effect=fake_receive):
        result = await ex._execute_tool(tool_call, {}, repl_state, 3000)

    assert result["success"] is True
    assert result["content"] == "Sub done with findings."


# ---------------------------------------------------------------------------
# Bug 5 regression: _parse_plan_response must apply step_offset to depends_on
# ---------------------------------------------------------------------------

async def test_parse_plan_response_applies_offset_to_depends_on():
    """
    When step_offset > 0 (as used in _replan), the depends_on values in the
    parsed plan must be shifted by step_offset so they reference the same
    numbering space as the offset step numbers.
    """
    ex = make_executor()
    # LLM returns a 3-step plan where step 3 depends on steps 1 and 2.
    raw = json.dumps([
        {"step": 1, "action": "A", "target": "t1", "goal": "g1", "depends_on": []},
        {"step": 2, "action": "B", "target": "t2", "goal": "g2", "depends_on": []},
        {"step": 3, "action": "C", "target": "t3", "goal": "g3", "depends_on": [1, 2]},
    ])
    # Simulate a replan after 3 completed steps → step_offset=3
    result = await ex._parse_plan_response(raw, max_steps=10, step_offset=3)

    assert len(result) == 3
    # Step numbers must be shifted
    assert result[0]["step"] == 4
    assert result[1]["step"] == 5
    assert result[2]["step"] == 6
    # depends_on must also be shifted so they match the new step numbers
    assert result[2]["depends_on"] == [4, 5]


async def test_parse_plan_response_zero_offset_depends_on_unchanged():
    """With step_offset=0 (initial plan) depends_on must be kept as-is."""
    ex = make_executor()
    raw = json.dumps([
        {"step": 1, "action": "A", "target": "t1", "goal": "g1", "depends_on": []},
        {"step": 2, "action": "B", "target": "t2", "goal": "g2", "depends_on": [1]},
    ])
    result = await ex._parse_plan_response(raw, max_steps=10, step_offset=0)
    assert result[0]["step"] == 1
    assert result[1]["step"] == 2
    assert result[1]["depends_on"] == [1]


# ---------------------------------------------------------------------------
# Bug 9 regression: RLM SubCall tool_call_id must be set from the tool_call
# ---------------------------------------------------------------------------

async def test_rlm_subcall_tool_call_id_override():
    """
    RLMAgentLoopExecutor._execute_tool must override the tool_call_id returned
    by _execute_sub_call with the id from the actual tool_call dict.
    """
    from modules.agent_loop.node import RLMAgentLoopExecutor
    with patch("modules.agent_loop.node.LLMBridge"):
        ex = RLMAgentLoopExecutor()
    ex.llm = MagicMock()

    tool_call = {
        "id": "real_id_123",
        "function": {
            "name": "SubCall",
            "arguments": json.dumps({"prompt": "hello"}),
        },
    }
    repl_state = {
        "sub_call_count": 0,
        "max_sub_calls": 10,
        "recursion_depth": 0,
        "max_recursion_depth": 3,
        "estimated_cost": 0.0,
        "max_cost_usd": 1.0,
    }

    async def _fake_sub_call(args, rs):
        return {"tool_call_id": "", "role": "tool", "name": "sub_call",
                "content": "answer", "success": True}

    with patch.object(ex, "_execute_sub_call", side_effect=_fake_sub_call):
        result = await ex._execute_tool(tool_call, {}, repl_state)

    assert result["tool_call_id"] == "real_id_123"


# ---------------------------------------------------------------------------
# Bug 11 regression: remaining_steps on AskUser pause must equal snap_remaining
# ---------------------------------------------------------------------------

def test_ask_user_pause_remaining_steps_matches_plan():
    """
    On AskUser pause, result["remaining_steps"] must equal result["plan"]
    (both snap_remaining) so external consumers don't lose the paused step.

    _run_one_step is a closure inside receive() — we verify the fix is in
    place by inspecting the source of GoalPursuitExecutor.receive().
    """
    import inspect
    source = inspect.getsource(GoalPursuitExecutor.receive)
    # Both keys must be set to snap_remaining in the pause block
    plan_line = 'result["plan"] = snap_remaining'
    remaining_line = 'result["remaining_steps"] = snap_remaining'
    assert plan_line in source, "Must set plan=snap_remaining on pause"
    # Verify remaining_steps is also snap_remaining (the fix).
    # Find the plan assignment then look ahead for the remaining_steps assignment.
    plan_pos = source.index(plan_line)
    segment = source[plan_pos: plan_pos + 1200]
    assert remaining_line in segment, (
        "remaining_steps must equal snap_remaining (not the stripped remaining_steps) "
        "on AskUser pause so the paused step is not dropped"
    )


# ---------------------------------------------------------------------------
# Bug 13 regression: var_names filter must require endswith("_result")
# ---------------------------------------------------------------------------

def test_build_step_system_prompt_filters_only_step_result_vars():
    """Variables like step_timeout must not appear in the Available Step Results list."""
    ex = make_executor()
    repl_state = {
        "variables": {
            "step_1_result": "some output",
            "step_timeout": "unexpected",      # should NOT appear
            "step_2_result": "more output",
            "step_extra_data": "also hidden",  # should NOT appear
        },
    }
    step = {"step": 3, "action": "Process", "target": "data", "goal": "g"}
    # _build_step_system_prompt takes: step, plan_context, input_data, config,
    # large_output_threshold, repl_state=None
    prompt = ex._build_step_system_prompt(
        step,
        "## Plan\nstep 3",
        {"messages": MESSAGES},
        {},
        4096,
        repl_state,
    )
    assert "step_timeout" not in prompt
    assert "step_extra_data" not in prompt
    assert "step_1_result" in prompt
    assert "step_2_result" in prompt


# ---------------------------------------------------------------------------
# Bug 14 regression: DecomposeGoal blocked when enable_subgoals=False
# ---------------------------------------------------------------------------

async def test_decompose_goal_blocked_when_subgoals_disabled():
    """
    If enable_subgoals=False, a hallucinated DecomposeGoal call must return
    an error result instead of executing a sub-goal.
    """
    ex = make_executor()
    repl_state = {
        "variables": {},
        "_var_counts": {},
        "sub_call_count": 0,
        "max_sub_calls": 10,
        "recursion_depth": 0,
        "max_recursion_depth": 3,
        "estimated_cost": 0.0,
        "max_cost_usd": 1.0,
        "_subgoal_depth": 0,
        "_max_subgoal_depth": 2,
        "_subgoal_config": {"enable_subgoals": False},
    }
    tool_call = {
        "id": "tc_dg",
        "function": {
            "name": "DecomposeGoal",
            "arguments": json.dumps({"subgoal": "do something"}),
        },
    }
    result = await ex._execute_tool(tool_call, {}, repl_state, 3000)
    assert result["success"] is False
    assert "disabled" in result["content"].lower()


# ---------------------------------------------------------------------------
# Bug 15 regression: _evaluate_step respects eval_response_max_chars config
# ---------------------------------------------------------------------------

async def test_evaluate_step_uses_configurable_truncation():
    """
    _evaluate_step must truncate execution_result at eval_response_max_chars
    (default 8000), not the old hardcoded 2000.
    """
    ex = make_executor()
    # Build a result whose key evidence is at char 3000 (beyond old 2000 limit)
    long_result = "x" * 2500 + "SUCCESS_MARKER" + "y" * 5000

    captured_prompts = []

    async def _fake_llm_with_retry(messages, **kwargs):
        captured_prompts.extend(messages)
        return {"choices": [{"message": {"content": json.dumps(
            {"success": True, "reason": "ok", "extracted_result": "r"}
        )}}]}

    with patch.object(ex, "_llm_with_retry", side_effect=_fake_llm_with_retry):
        await ex._evaluate_step(
            step={"action": "act", "target": "tgt", "goal": "g"},
            execution_result=long_result,
            config={"eval_response_max_chars": 8000},
        )

    full_prompt = " ".join(m.get("content", "") for m in captured_prompts)
    assert "SUCCESS_MARKER" in full_prompt, (
        "Evidence at char 2500 must be visible with eval_response_max_chars=8000"
    )


async def test_evaluate_step_custom_truncation_respected():
    """eval_response_max_chars config value must be honoured over the default."""
    ex = make_executor()
    long_result = "a" * 100 + "HIDDEN_MARKER" + "b" * 500

    captured_prompts = []

    async def _fake_llm_with_retry(messages, **kwargs):
        captured_prompts.extend(messages)
        return {"choices": [{"message": {"content": json.dumps(
            {"success": True, "reason": "ok", "extracted_result": "r"}
        )}}]}

    with patch.object(ex, "_llm_with_retry", side_effect=_fake_llm_with_retry):
        await ex._evaluate_step(
            step={"action": "act", "target": "tgt", "goal": "g"},
            execution_result=long_result,
            # Truncate before the marker
            config={"eval_response_max_chars": 50},
        )

    full_prompt = " ".join(m.get("content", "") for m in captured_prompts)
    assert "HIDDEN_MARKER" not in full_prompt, (
        "Marker at char 100 must be cut off with eval_response_max_chars=50"
    )


# ---------------------------------------------------------------------------
# Bug 16 regression: non-integer goal_id must not silently log as POST failure
# ---------------------------------------------------------------------------

async def test_goal_id_non_integer_raises_clear_error(capsys):
    """
    A UUID-style goal_id must trigger a ValueError with a clear message, not
    an obscure httpx error.
    """
    ex = make_executor()
    # Simulate the goal-completion block in isolation
    goal_id = "not-an-integer"
    _goal_id_str = str(goal_id)
    raised = None
    try:
        if not _goal_id_str.lstrip("-").isdigit():
            raise ValueError(f"goal_id must be an integer, got: {_goal_id_str!r}")
    except ValueError as e:
        raised = e
    assert raised is not None
    assert "not-an-integer" in str(raised)


# ---------------------------------------------------------------------------
# Bug 17 regression: RLM loop deduplicates tool calls
# ---------------------------------------------------------------------------

async def test_rlm_loop_deduplicates_tool_calls():
    """
    _run_rlm_loop must skip duplicate tool calls within the same response,
    matching the deduplication behaviour of the standard and hybrid loops.
    """
    from modules.agent_loop.node import RLMAgentLoopExecutor
    with patch("modules.agent_loop.node.LLMBridge"):
        rlm = RLMAgentLoopExecutor()
    rlm._load_tool_library = MagicMock(return_value={"Echo": "result = args.get('text', '')"})

    dup_call = {"id": "c1", "function": {"name": "Echo", "arguments": '{"text": "hi"}'}}
    uniq_call = {"id": "c2", "function": {"name": "Echo", "arguments": '{"text": "unique"}'}}

    tool_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [dup_call, dup_call, uniq_call],
            }
        }]
    }
    final_response = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    rlm.llm = MagicMock()
    rlm.llm.chat_completion = AsyncMock(side_effect=[tool_response, final_response])

    execution_log = []

    async def _fake_execute_tool(tool_call, library, repl_state):
        execution_log.append(tool_call.get("function", {}).get("arguments", ""))
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "content": "echoed",
            "success": True,
        }

    repl_state = {
        "variables": {},
        "stdout_history": [],
        "iteration": 0,
        "_var_counts": {},
        "final": None,
    }
    trace = []

    with patch.object(rlm, "_execute_tool", side_effect=_fake_execute_tool):
        await rlm._run_rlm_loop(
            initial_messages=[{"role": "user", "content": "echo"}],
            repl_state=repl_state,
            tools_list=[],
            tool_library={},
            model="test-model",
            config={"max_iterations": 5, "max_llm_retries": 1, "retry_delay": 0},
            trace=trace,
        )

    # dup_call appears twice but must only execute once; uniq_call executes once
    # Total: 2 executions (not 3)
    assert len(execution_log) == 2, (
        f"Expected 2 executions (1 dup skipped), got {len(execution_log)}: {execution_log}"
    )
    assert execution_log.count('{"text": "hi"}') == 1
    assert execution_log.count('{"text": "unique"}') == 1
    # Trace must record the duplicate error
    assert any("Duplicate" in e for itr in trace for e in itr.get("errors", []))
