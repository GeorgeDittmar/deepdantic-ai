"""Critic result handling extracted from ``agent.py``.

Extracted so the critic feedback loop (status transitions, checkpoint events,
objective patching) can be unit-tested in isolation.

Design
------
Each method receives only the state it needs: the ``TaskItem`` to update and
the ``TaskQAResult`` from the critic. Checkpoint events are emitted via
callbacks that delegate back to ``DeepAgent``'s checkpointing methods.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger
from pydantic_ai import RunContext

from pydantask.models import TaskItem, TaskQAResult, TaskStatus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CriticHandler:
    """Handle critic QA results for a single task.

    Each method is a thin wrapper around task state mutation and checkpoint
    event emission. Callbacks delegate back to ``DeepAgent`` for checkpointing.

    Typical usage::

        handler = CriticHandler(
            record_event_cb=lambda et, p: ...,
            record_task_status_event_cb=lambda tid, st, **kw: ...,
        )
        # handler.handle_critic_result can be called directly
    """

    def __init__(
        self,
        record_event_cb: Callable,
        record_task_status_event_cb: Callable,
    ) -> None:
        """Initialize CriticHandler.

        Args:
            record_event_cb: Callback for checkpoint events (delegates to
                ``DeepAgent._record_event``).
            record_task_status_event_cb: Callback for status-change events
                (delegates to ``DeepAgent._record_task_status_event``).
        """
        self._record_event_cb = record_event_cb
        self._record_task_status_event_cb = record_task_status_event_cb

        self._tools: dict[str, Callable] = {
            "handle_critic_result": self.handle_critic_result,
        }

    @property
    def handle_critic_result(self) -> Callable:
        """Bound ``handle_critic_result`` method."""
        return self._tools["handle_critic_result"]

    async def handle_critic_result(self, task: TaskItem, review: TaskQAResult) -> None:
        """Apply the critic's QA result to a task and emit checkpoint events.

        Status transitions:
        - ``passed=True`` → ``COMPLETED``
        - ``passed=False`` and ``attempt_count >= max_attempts`` → ``FAILED``
        - ``passed=False`` and ``attempt_count < max_attempts`` → ``RERUN``
          (also patches the ``sub_task_objective`` with feedback).

        Args:
            task: The ``TaskItem`` that was reviewed.
            review: The critic's ``TaskQAResult`` with pass/fail verdict.
        """
        task.attempt_count += 1
        task.task_feedback = review

        await self._record_event_cb(
            "critic_feedback",
            {
                "task_id": task.task_id,
                "feedback": review.model_dump(mode="json"),
                "attempt_count": task.attempt_count,
            },
        )

        if review.passed:
            task.status = TaskStatus.COMPLETED
            task.error_msg = None
            await self._record_task_status_event_cb(task.task_id, task.status)
            return

        if task.attempt_count >= task.max_attempts:
            task.status = TaskStatus.FAILED
            task.error_msg = (
                f"Max retries reached ({task.attempt_count}/{task.max_attempts})."
            )
            await self._record_task_status_event_cb(
                task.task_id, task.status, error_msg=task.error_msg
            )
            return

        task.status = TaskStatus.RERUN
        task.error_msg = None
        task.sub_task_objective = f"{task.sub_task_objective}\n\nPrevious attempt failed review; feedback: {review.reasoning}"
        await self._record_task_status_event_cb(task.task_id, task.status)
        await self._record_event_cb(
            "task_patched",
            {
                "task_id": task.task_id,
                "sub_task_objective": task.sub_task_objective,
            },
        )
