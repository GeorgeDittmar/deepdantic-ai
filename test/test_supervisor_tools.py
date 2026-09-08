"""Unit tests for the extracted SupervisorTools class.

Each test exercises the tool methods directly on a SupervisorTools instance,
without going through DeepAgent delegation.  This validates that the extracted
logic is correct independent of the host class.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai import RunContext

from pydantask.agents.supervisor_tools import SupervisorTools
from pydantask.models import TaskItem, TaskQAResult, TaskStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_task(task_id: int = 1, **kwargs) -> TaskItem:
    """Build a minimal TaskItem with sensible defaults."""
    return TaskItem(
        task_id=task_id,
        overall_objective=kwargs.pop("objective", "obj"),
        sub_task_objective=kwargs.pop("sub_task_objective", "x"),
        capability=kwargs.pop("capability", "w"),
        sub_task_dependencies=kwargs.pop("sub_task_dependencies", []),
        metadata=kwargs.pop("metadata", {}),
        parameters=kwargs.pop("parameters", {}),
        status=kwargs.pop("status", TaskStatus.READY),
        is_final=kwargs.pop("is_final", False),
        max_attempts=kwargs.pop("max_attempts", 3),
        task_feedback=kwargs.pop("task_feedback", None),
        **kwargs,
    )


def _make_runtime(
    plan: dict | None = None,
    objective: str = "test objective",
    next_task_id: int = 1,
) -> SimpleNamespace:
    """Build a lightweight RuntimeState-like namespace for testing."""
    state = SimpleNamespace(
        plan=plan if plan is not None else {},
        objective=objective,
        next_task_id=next_task_id,
        checkpoint_recorder=None,
    )
    state.record_event = state.checkpoint_recorder.record if state.checkpoint_recorder else AsyncMock()
    state.record_task_status_event = (
        state.checkpoint_recorder.record_task_status_event
        if state.checkpoint_recorder
        else AsyncMock()
    )
    return state


def _make_tools(**kwargs):
    """Build a SupervisorTools with default context_resolver."""
    return SupervisorTools(
        context_resolver=lambda ctx: ctx.deps,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# add_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_task_creates_task_and_increments_id():
    runtime = _make_runtime(plan={}, objective="obj", next_task_id=1)

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    task_id = await tools.add_task(
        ctx,
        sub_task_objective="do the thing",
        capability="worker_agent",
        dependencies=[99],
        metadata={"key": "val"},
    )

    assert task_id == 1
    assert runtime.next_task_id == 2
    assert runtime.plan[1].task_id == 1
    assert runtime.plan[1].sub_task_objective == "do the thing"
    assert runtime.plan[1].capability == "worker_agent"
    assert runtime.plan[1].sub_task_dependencies == [99]
    assert runtime.plan[1].metadata == {"key": "val"}
    assert runtime.plan[1].status == TaskStatus.READY


@pytest.mark.asyncio
async def test_add_task_emits_checkpoint_event():
    recorder = MagicMock()
    recorder.record = AsyncMock()

    runtime = _make_runtime(plan={})
    runtime.checkpoint_recorder = recorder
    runtime.record_event = recorder.record
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    await tools.add_task(
        ctx,
        sub_task_objective="checkpoint",
        capability="worker_agent",
    )

    assert recorder.record.called
    call_args = recorder.record.call_args
    assert call_args[0][0] == "task_added"
    assert call_args[0][1]["task"]["task_id"] == 1
    assert call_args[0][1]["next_task_id"] == 2


@pytest.mark.asyncio
async def test_add_task_without_deps_or_metadata():
    runtime = _make_runtime(plan={}, next_task_id=5)

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    task_id = await tools.add_task(
        ctx,
        sub_task_objective="simple task",
        capability="research_agent",
    )

    assert task_id == 5
    assert runtime.plan[5].sub_task_dependencies == []
    assert runtime.plan[5].metadata == {}


# ---------------------------------------------------------------------------
# cancel_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_task_sets_status():
    runtime = _make_runtime(plan={1: _make_task(1)})
    runtime.record_event = AsyncMock()
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    msg = await tools.cancel_task(ctx, task_id=1, reason="not needed")

    assert "cancelled" in msg
    assert runtime.plan[1].status == TaskStatus.CANCELLED
    assert runtime.plan[1].error_msg == "not needed"


@pytest.mark.asyncio
async def test_cancel_task_missing_id():
    runtime = _make_runtime(plan={})
    runtime.record_event = AsyncMock()
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    msg = await tools.cancel_task(ctx, task_id=99, reason="missing")

    assert "not found" in msg


# ---------------------------------------------------------------------------
# patch_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_task_updates_fields():
    runtime = _make_runtime(
        plan={
            1: _make_task(1, sub_task_dependencies=[1]),
        }
    )
    runtime.record_event = AsyncMock()
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    msg = await tools.patch_task(
        ctx,
        task_id=1,
        sub_task_objective="new objective",
        dependencies=[10, 20],
    )

    assert "updated successfully" in msg
    assert runtime.plan[1].sub_task_objective == "new objective"
    assert runtime.plan[1].sub_task_dependencies == [10, 20]


@pytest.mark.asyncio
async def test_patch_task_missing_task():
    runtime = _make_runtime(plan={})
    runtime.record_event = AsyncMock()
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    msg = await tools.patch_task(ctx, task_id=99, sub_task_objective="x")

    assert "not found" in msg


@pytest.mark.asyncio
async def test_patch_task_with_parameters():
    runtime = _make_runtime(
        plan={
            1: _make_task(1, parameters={"a": 1}),
        }
    )
    runtime.record_event = AsyncMock()
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    msg = await tools.patch_task(
        ctx, task_id=1, parameters={"b": 2},
    )

    assert runtime.plan[1].parameters == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# mark_final_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_final_task_sets_flag_and_clears_others():
    runtime = _make_runtime(
        plan={
            1: _make_task(1, is_final=True),
            2: _make_task(2, is_final=False),
            3: _make_task(3, is_final=False),
        }
    )
    runtime.record_event = AsyncMock()
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    msg = await tools.mark_final_task(ctx, task_id=3, reason="unit")

    assert "marked as final" in msg
    assert runtime.plan[1].is_final is False
    assert runtime.plan[2].is_final is False
    assert runtime.plan[3].is_final is True


@pytest.mark.asyncio
async def test_mark_final_task_missing_id():
    runtime = _make_runtime(plan={})
    runtime.record_event = AsyncMock()
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    msg = await tools.mark_final_task(ctx, task_id=99)

    assert "No task with id" in msg


# ---------------------------------------------------------------------------
# update_task_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_status_changes_status():
    runtime = _make_runtime(
        plan={1: _make_task(1, status=TaskStatus.READY)}
    )
    runtime.record_event = AsyncMock()
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    msg = await tools.update_task_status(ctx, task_id=1, status=TaskStatus.COMPLETED)

    assert "now" in msg
    assert runtime.plan[1].status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_update_task_status_missing():
    runtime = _make_runtime(plan={})
    runtime.record_event = AsyncMock()
    runtime.record_task_status_event = AsyncMock()

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    msg = await tools.update_task_status(ctx, task_id=99, status=TaskStatus.COMPLETED)

    assert "No task with" in msg


# ---------------------------------------------------------------------------
# view_qa_report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_view_qa_report_returns_json():
    runtime = _make_runtime(
        plan={
            1: _make_task(
                1,
                task_feedback=TaskQAResult(task_id=1, passed=True, reasoning="looks good"),
            )
        }
    )

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    report = await tools.view_qa_report(ctx, task_id=1)

    assert '"passed": true' in report
    assert '"reasoning": "looks good"' in report
    # Verify metadata was updated
    assert runtime.plan[1].metadata.get("qa", {}).get("report_viewed") is True


@pytest.mark.asyncio
async def test_view_qa_report_no_feedback():
    runtime = _make_runtime(
        plan={1: _make_task(1)}
    )

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    report = await tools.view_qa_report(ctx, task_id=1)

    assert "No QA feedback" in report


@pytest.mark.asyncio
async def test_view_qa_report_missing_task():
    runtime = _make_runtime(plan={})

    tools = _make_tools()
    ctx = SimpleNamespace(deps=runtime)

    report = await tools.view_qa_report(ctx, task_id=99)

    assert "No task with id 99" in report


# ---------------------------------------------------------------------------
# next_task_id property
# ---------------------------------------------------------------------------


def test_next_task_id_reflects_runtime():
    runtime = _make_runtime(plan={}, next_task_id=42)

    tools = _make_tools()
    # next_task_id lives on RuntimeState (not SupervisorTools) since extraction
    assert runtime.next_task_id == 42
