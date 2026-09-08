"""Supervisor tools for DAG mutation.

Extracted from agent.py so each tool can be unit-tested in isolation.

Design
------
Each tool factory captures *counter* (shared mutable next-task-id) and
*context_resolver* (called at runtime to extract plan, objective, and
checkpoint callbacks from the ``RunContext``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from loguru import logger
from pydantic_ai import RunContext

from pydantask.models import TaskItem, TaskStatus


# ---------------------------------------------------------------------------
# Factory closures — each returns a tool bound to *counter* + resolver
# ---------------------------------------------------------------------------

def _make_add_task_callback(
    context_resolver: Callable[[Any], Any],
) -> Callable:
    """Return an ``add_task`` bound to *context_resolver*."""

    async def add_task(
        ctx: RunContext[Any],
        sub_task_objective: str,
        capability: str,
        dependencies: list[int] | None = None,
        metadata: dict | None = None,
        parameters: dict | None = None,
    ) -> int:
        """Tool: Add Task.

        Create and register a new ``TaskItem`` in the current plan/DAG when
        more work is required to achieve the overall objective.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            sub_task_objective: Natural-language objective for the new task.
            capability: Name of the capability / sub-agent that should execute
                this task.
            dependencies: Optional list of task IDs that must complete
                successfully before this task can run.
            metadata: Optional free-form metadata dictionary attached to the task.

        Returns:
            The integer ``task_id`` assigned to the newly created task.
        """
        s = context_resolver(ctx)
        new_id = s.next_task_id
        s.next_task_id += 1
        task = TaskItem(
            task_id=new_id,
            overall_objective=s.objective,
            sub_task_objective=sub_task_objective,
            capability=capability,
            sub_task_dependencies=dependencies or [],
            metadata=metadata or {},
            parameters=parameters or {},
            status=TaskStatus.READY,
        )
        s.plan[new_id] = task
        await s.record_event(
            "task_added",
            {
                "task": task.model_dump(mode="json"),
                "next_task_id": s.next_task_id,
            },
        )
        return new_id

    return add_task


def _make_cancel_task_callback(
    context_resolver: Callable[[Any], Any],
) -> Callable:
    """Return a ``cancel_task`` bound to *context_resolver*."""

    async def cancel_task(
        ctx: RunContext[Any], task_id: int, reason: str
    ) -> str:
        """Tool: Cancel Task.

        Mark a task as ``CANCELLED`` when it is no longer relevant or when
        a failure in an upstream dependency makes it impossible to complete.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            task_id: Identifier of the task to cancel.
            reason: Human-readable explanation for the cancellation.
        """
        s = context_resolver(ctx)
        plan = s.plan
        if task_id in plan:
            task = plan[task_id]
            task.status = TaskStatus.CANCELLED
            task.error_msg = reason
            await s.record_task_status_event(
                task_id,
                TaskStatus.CANCELLED,
                reason=reason,
                error_msg=reason,
            )
            return f"Task {task_id} cancelled. Reason: {reason}"
        return f"Error: Task {task_id} not found."

    return cancel_task


def _make_patch_task_callback(
    context_resolver: Callable[[Any], Any],
) -> Callable:
    """Return a ``patch_task`` bound to *context_resolver*."""

    async def patch_task(
        ctx: RunContext[Any],
        task_id: int,
        sub_task_objective: Optional[str] = None,
        capability: Optional[str] = None,
        dependencies: Optional[list[int]] = None,
        parameters: dict | None = None,
    ) -> str:
        """Tool: Patch Task.

        Update an existing task's objective and/or dependency list in-place.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            task_id: Identifier of the task to modify.
            sub_task_objective: New sub-task objective, if changing.
            capability: New capability to use, if changing.
            dependencies: Updated list of dependency IDs, if changing.
        """
        s = context_resolver(ctx)
        plan = s.plan
        task = plan.get(task_id)
        if not task:
            return "Task not found."

        payload: Dict[str, Any] = {"task_id": task_id}

        if sub_task_objective:
            task.sub_task_objective = sub_task_objective
            payload["sub_task_objective"] = task.sub_task_objective
        if dependencies is not None:
            task.sub_task_dependencies = dependencies
            payload["dependencies"] = task.sub_task_dependencies

        if capability is not None:
            task.capability = capability
            payload["capability"] = task.capability

        if parameters is not None:
            if not isinstance(getattr(task, "parameters", None), dict):
                task.parameters = {}
            if not isinstance(parameters, dict):
                return "Error: 'parameters' must be a dict."
            task.parameters.update(parameters)
            payload["parameters"] = parameters

        if len(payload) > 1:
            await s.record_event("task_patched", payload)

        return f"Task {task_id} updated successfully."

    return patch_task


def _make_mark_final_task_callback(
    context_resolver: Callable[[Any], Any],
) -> Callable:
    """Return a ``mark_final_task`` bound to *context_resolver*."""

    async def mark_final_task(
        ctx: RunContext[Any],
        task_id: int,
        reason: str | None = None,
    ) -> str:
        """Tool: Mark Final Task.

        Mark exactly one task as the final deliverable for the run.

        The invariant enforced is: at most one task has ``is_final=True``.
        """
        s = context_resolver(ctx)
        plan = s.plan
        if task_id not in plan:
            return f"Error: No task with id {task_id} found in plan."

        for t in plan.values():
            t.is_final = False

        plan[task_id].is_final = True

        payload: Dict[str, Any] = {"task_id": task_id}
        if reason:
            payload["reason"] = reason
        await s.record_event("final_task_set", payload)

        return f"Task {task_id} marked as final."

    return mark_final_task


def _make_update_task_status_callback(
    context_resolver: Callable[[Any], Any],
) -> Callable:
    """Return an ``update_task_status`` bound to *context_resolver*."""

    async def update_task_status(
        ctx: RunContext[Any], task_id: int, status: TaskStatus
    ) -> str:
        """Tool: Update Task Status.

        Primarily used by the supervisor to transition a task between states
        (e.g. to ``READY`` or ``COMPLETED``) once dependencies are met or QA
        has passed.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            task_id: Identifier of the task to update.
            status: New :class:`TaskStatus` value for the task.
        """
        s = context_resolver(ctx)
        plan = s.plan
        task = plan.get(task_id)
        if task is not None:
            task.status = status
            await s.record_task_status_event(task_id, status)
            return f"Status for {task_id} is now {status}."
        return f"Error: No task with {task_id} found in plan. Be sure task_id actually exists."

    return update_task_status


def _make_view_qa_report_callback(
    context_resolver: Callable[[Any], Any],
) -> Callable:
    """Return a ``view_qa_report`` bound to *context_resolver*."""

    async def view_qa_report(
        ctx: RunContext[Any], task_id: int
    ) -> str:
        """Tool: View QA Report.

        Return the full serialized QA report for a specific task, if one is
        available.

        Args:
            ctx: ``RunContext`` carrying the current ``RuntimeState``.
            task_id: Identifier of the task whose QA report should be viewed.

        Returns:
            A JSON-formatted string representation of the stored
            :class:`TaskQAResult`, or a message describing why no report is
            available.
        """
        s = context_resolver(ctx)
        plan = s.plan
        task = plan.get(task_id)
        logger.info(f"Checking QA Report for task: {task_id}")
        if task is None:
            return f"No task with id {task_id}."

        fb = getattr(task, "task_feedback", None)
        if fb is None:
            return f"No QA feedback found for task {task_id}."
        task.metadata.setdefault("qa", {})
        task.metadata["qa"]["report_viewed"] = True

        return fb.model_dump_json(indent=2)

    return view_qa_report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SupervisorTools:
    """Tool methods for the supervisor agent (DAG mutation).

    Each method is a thin wrapper around shared mutable state (the plan dict).
    The plan, objective, and checkpoint callbacks are extracted at call time
    via a *context_resolver* — a callable that receives a ``RunContext`` and
    returns an object exposing ``plan``, ``objective``, ``next_task_id``,
    ``record_event``, and ``record_task_status_event``.

    Typical usage::

        tools = SupervisorTools(
            context_resolver=lambda ctx: ctx.deps,
        )
        # tools.add_task is a callable ready for Agent(tool=...) registration
    """

    def __init__(
        self,
        context_resolver: Callable[[Any], Any],
    ) -> None:
        """Initialize SupervisorTools.

        Args:
            context_resolver: Called with each ``RunContext`` to extract the
                values the tools need: ``.plan``, ``.objective``,
                ``.record_event``, ``.record_task_status_event``.
        """
        self._resolver = context_resolver

        self._tools: dict[str, Callable] = {
            "add_task": _make_add_task_callback(context_resolver),
            "cancel_task": _make_cancel_task_callback(context_resolver),
            "patch_task": _make_patch_task_callback(context_resolver),
            "mark_final_task": _make_mark_final_task_callback(context_resolver),
            "update_task_status": _make_update_task_status_callback(context_resolver),
            "view_qa_report": _make_view_qa_report_callback(context_resolver),
        }

    @property
    def add_task(self) -> Callable:
        """Bound ``add_task`` tool method."""
        return self._tools["add_task"]

    @property
    def cancel_task(self) -> Callable:
        """Bound ``cancel_task`` tool method."""
        return self._tools["cancel_task"]

    @property
    def patch_task(self) -> Callable:
        """Bound ``patch_task`` tool method."""
        return self._tools["patch_task"]

    @property
    def mark_final_task(self) -> Callable:
        """Bound ``mark_final_task`` tool method."""
        return self._tools["mark_final_task"]

    @property
    def update_task_status(self) -> Callable:
        """Bound ``update_task_status`` tool method."""
        return self._tools["update_task_status"]

    @property
    def view_qa_report(self) -> Callable:
        """Bound ``view_qa_report`` tool method."""
        return self._tools["view_qa_report"]
