"""Task execution logic extracted from ``agent.py``.

Extracted so the task execution loop (concurrent scheduling, single-task
execution with retry-on-overflow, output coercion) can be unit-tested in
isolation from the rest of ``DeepAgent``.

Design
------
* ``context_resolver`` — called at runtime to extract ``plan``,
  ``objective``, ``capability_registry``, and other values from the
  ``RunContext`` / ``RuntimeState``.
* ``record_event_cb`` / ``record_task_status_event_cb`` /
  ``record_metadata_append_cb`` — callbacks that delegate back to
  ``DeepAgent``'s checkpointing methods (kept on agent.py since they need
  the ``_checkpoint_recorder``).
* ``coerce_output_cb`` — callback that delegates back to
  ``DeepAgent._coerce_output_to_task_result`` for output normalization and
  file ingestion.
* ``record_task_result_cb`` — callback that delegates back to
  ``DeepAgent._record_task_result`` for checkpoint-payload persistence and
  truncation.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from asyncio import TaskGroup
from datetime import datetime
from http import HTTPStatus
from httpx import AsyncClient, HTTPStatusError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_ai.usage import UsageLimits
from tenacity import (
    wait_exponential,
    retry_if_exception_type,
    stop_after_attempt,
)

from pydantask.capabilities.runner_v2 import CapabilityRunner
from pydantask.models import (
    ArtifactRef,
    RuntimeState,
    TaskItem,
    TaskResult,
    TaskRunDeps,
    TaskStatus,
)
from pydantask.observe.tracing import traced

EVENT_RESULT_DETAIL_TRUNCATION = 4_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TaskExecutor:
    """Execute sub-agent tasks: concurrent scheduling + single-task execution.

    Each method delegates to internal helpers. Contextual state (plan,
    capability_registry, objective) is extracted at call time via a
    *context_resolver*.  Checkpoint operations are performed via callbacks
    that reference back into ``DeepAgent``.

    Typical usage::

        executor = TaskExecutor(
            context_resolver=lambda ctx: ctx,
            record_event_cb=lambda et, p: ...,
            record_task_status_event_cb=lambda tid, st, **kw: ...,
            record_metadata_append_cb=lambda tid, k, v: ...,
            coerce_output_cb=lambda step, out, rs: ...,
            record_task_result_cb=lambda step: ...,
        )
    """

    def __init__(
        self,
        context_resolver: Callable[[Any], Any],
        capability_registry: Dict,
        record_event_cb: Callable,
        record_task_status_event_cb: Callable,
        record_metadata_append_cb: Callable,
        coerce_output_cb: Callable,
        record_task_result_cb: Callable,
    ) -> None:
        """Initialize TaskExecutor.

        Args:
            context_resolver: Called to extract runtime values (plan,
                objective, capability_registry, etc.).
            capability_registry: Mapping of capability names to
                ``CapabilityDescription`` instances.
            record_event_cb: Callback for checkpoint events (delegates to
                ``DeepAgent._record_event``).
            record_task_status_event_cb: Callback for status-change events
                (delegates to ``DeepAgent._record_task_status_event``).
            record_metadata_append_cb: Callback for metadata-append events
                (delegates to ``DeepAgent._record_metadata_append``).
            coerce_output_cb: Callback that normalizes capability output into
                a ``TaskResult`` (delegates to ``DeepAgent._coerce_output_to_task_result``).
            record_task_result_cb: Callback that persists a task result to
                checkpoint (delegates to ``DeepAgent._record_task_result``).
        """
        self._resolver = context_resolver
        self._capability_registry = capability_registry
        self._record_event_cb = record_event_cb
        self._record_task_status_event_cb = record_task_status_event_cb
        self._record_metadata_append_cb = record_metadata_append_cb
        self._coerce_output_cb = coerce_output_cb
        self._record_task_result_cb = record_task_result_cb
        # Callback for running a single task (used by _execute_ready_tasks).
        # Defaults to self.execute but can be overridden for testing.
        self._run_task_cb: Callable | None = None

        self._tools: dict[str, Callable] = {
            "execute": self.execute,
            "execute_ready_tasks": self._execute_ready_tasks,
            "is_context_limit_error": self._is_context_limit_error,
            "build_resume_prompt": self._build_resume_prompt,
            "truncate_text": self._truncate_text,
            "remaining_token_budget": self._remaining_token_budget,
            "make_usage_limits": self._make_usage_limits,
            "extract_total_tokens": self._extract_total_tokens,
            "accumulate_usage": self._accumulate_usage,
        }

    @property
    def run_task_cb(self) -> Callable:
        """Callback for running a single task (defaults to self.execute)."""
        return self._run_task_cb or self.execute

    @property
    def execute(self) -> Callable:
        """Bound ``execute`` method — runs a single task."""
        return self._tools["execute"]

    @property
    def execute_ready_tasks(self) -> Callable:
        """Bound ``_execute_ready_tasks`` method — runs ready tasks concurrently."""
        return self._tools["execute_ready_tasks"]

    @property
    def is_context_limit_error(self) -> Callable:
        """Bound ``_is_context_limit_error`` heuristic checker."""
        return self._tools["is_context_limit_error"]

    @property
    def build_resume_prompt(self) -> Callable:
        """Bound ``_build_resume_prompt`` helper."""
        return self._tools["build_resume_prompt"]

    @property
    def truncate_text(self) -> Callable:
        """Bound ``_truncate_text`` helper."""
        return self._tools["truncate_text"]

    @property
    def remaining_token_budget(self) -> Callable:
        """Bound ``_remaining_token_budget`` helper."""
        return self._tools["remaining_token_budget"]

    @property
    def make_usage_limits(self) -> Callable:
        """Bound ``_make_usage_limits`` helper."""
        return self._tools["make_usage_limits"]

    @property
    def extract_total_tokens(self) -> Callable:
        """Bound ``_extract_total_tokens`` helper."""
        return self._tools["extract_total_tokens"]

    @property
    def accumulate_usage(self) -> Callable:
        """Bound ``_accumulate_usage`` helper."""
        return self._tools["accumulate_usage"]

    # -----------------------------------------------------------------------
    # Token usage helpers
    # -----------------------------------------------------------------------

    def _remaining_token_budget(self, runtime_state: RuntimeState) -> int | None:
        """Return remaining global token budget (best-effort), or None if unlimited.

        Note: Some unit tests construct ``DeepAgent`` without calling ``__init__``.
        Use ``getattr`` to avoid AttributeError in those scenarios.
        """
        budget = getattr(self, "token_budget", None)
        if budget is None:
            return None
        remaining = int(budget) - int(getattr(runtime_state, "tokens_used", 0) or 0)
        return max(0, remaining)

    def _make_usage_limits(self, **kwargs) -> UsageLimits | None:
        """Create a UsageLimits instance using only supported fields.

        pydantic-ai's UsageLimits has changed field names across versions.
        This helper filters kwargs by the actual constructor signature so we
        can safely pass token limits when available.
        """
        try:
            sig = inspect.signature(UsageLimits)
            allowed = {
                k: v for k, v in kwargs.items() if v is not None and k in sig.parameters
            }
            return UsageLimits(**allowed) if allowed else None
        except Exception:
            tcl = kwargs.get("tool_calls_limit")
            if tcl is not None:
                try:
                    return UsageLimits(tool_calls_limit=tcl)
                except Exception:
                    return None
            return None

    def _extract_total_tokens(self, run_result: Any) -> int | None:
        """Best-effort extraction of total token usage from a pydantic-ai result."""
        if run_result is None:
            return None

        usage = getattr(run_result, "usage", None)
        try:
            usage = usage() if callable(usage) else usage
        except Exception:
            usage = None

        if usage is None:
            return None

        if isinstance(usage, dict):
            for k in ("total_tokens", "total", "tokens", "all_tokens"):
                v = usage.get(k)
                if isinstance(v, (int, float)):
                    return int(v)
            return None

        for attr in ("total_tokens", "total", "tokens", "all_tokens"):
            v = getattr(usage, attr, None)
            if isinstance(v, (int, float)):
                return int(v)

        return None

    def _accumulate_usage(
        self, runtime_state: RuntimeState, run_result: Any, *, label: str
    ) -> None:
        """Accumulate usage into runtime_state.tokens_used (best-effort)."""
        total = self._extract_total_tokens(run_result)
        if total is None:
            return

        runtime_state.tokens_used = int(
            getattr(runtime_state, "tokens_used", 0) or 0
        ) + int(total)
        verbose = getattr(self, "verbose", False)
        if verbose:
            logger.info(
                f"Usage recorded ({label}): +{total} tokens; total_used={runtime_state.tokens_used}"
            )

    # -----------------------------------------------------------------------
    # Context limit / error helpers
    # -----------------------------------------------------------------------

    def _is_context_limit_error(self, exc: Exception) -> bool:
        """Heuristic detection of "context length exceeded" errors.

        Different providers/local gateways surface these differently (OpenAI-style
        400s, Anthropic "prompt too long", llama.cpp "context overflow", etc.).
        """
        msg = str(exc).lower()
        needles = [
            "context length",
            "maximum context",
            "max context",
            "prompt is too long",
            "too many tokens",
            "context overflow",
            "exceeds the context",
            "token limit",
        ]
        if any(n in msg for n in needles):
            return True

        if isinstance(exc, HTTPStatusError):
            try:
                data = exc.response.json()
            except Exception:
                data = None

            if exc.response.status_code in (400, 413):
                if data and isinstance(data, dict):
                    err = data.get("error") or {}
                    code = (err.get("code") or "").lower()
                    emsg = (err.get("message") or "").lower()
                    if "context" in code or "context" in emsg:
                        return True

        return False

    def _build_resume_prompt(self, step: TaskItem, error: Exception) -> str:
        """Build a minimal resume prompt after a context overflow.

        We intentionally keep this short; the sub-agent should reconstruct its
        progress using task metadata (scratch notes / checkpoints).
        """
        objective = self._get_objective()

        checkpoint = step.metadata.get("scratch_notes", "")
        checkpoint_preview = checkpoint
        if len(checkpoint_preview) > 6_000:
            checkpoint_preview = (
                checkpoint_preview[:6_000] + "\n...[checkpoint truncated]..."
            )

        return f"""
A previous attempt to execute this task failed due to context/window limits.

Task:
- task_id: {step.task_id}
- capability: {step.capability}
- sub_task_objective: {step.sub_task_objective}

Overall objective:
{objective}

Checkpoint / scratch notes saved so far (authoritative):
{checkpoint_preview if checkpoint_preview else '<none>'}

Recovery instructions (IMPORTANT):
- Continue the task from the checkpoint above.
- Keep responses concise. Avoid pasting large blobs.
- If you need prior task outputs, call `get_task_result(task_id=..., max_chars=6000)` (or smaller).
- If you need a quick targeted answer from another capability, call `consult_capability(capability=..., question=...)`.
- After each major step, call `append_scratch_note(note=...)` with a short checkpoint:
  "what I did" + "what I will do next" + "open questions".
- If you feel you're approaching the context limit again, STOP calling tools and output the best possible `TaskResult`.

Error that triggered recovery (for debugging only):
{str(error)}
"""

    def _truncate_text(self, text: str, max_chars: int | None) -> str:
        """Best-effort truncation helper to reduce prompt/tool output size."""
        if max_chars is None:
            return text
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text

        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        return (
            text[:head_chars]
            + f"\n\n...[TRUNCATED {len(text) - max_chars} chars; original_len={len(text)}]...\n\n"
            + text[-tail_chars:]
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_objective(self) -> str:
        """Return the overall objective for prompt construction."""
        # Try the executor's own objective attribute first (set from agent.py).
        objective = getattr(self, "objective", None)
        if objective is not None:
            return objective
        # Then try the resolver.
        ctx = self._resolver(None)  # type: ignore[arg-type]
        objective = getattr(ctx, "objective", None)
        if objective is not None:
            return objective
        return "unknown"

    def _get_capability_registry(self) -> dict:
        """Get capability registry."""
        return self._capability_registry

    def _dependencies_satisfied(self, step: TaskItem, ctx: RuntimeState) -> bool:
        """Check if all dependencies of a step are completed."""
        for dep_id in getattr(step, "sub_task_dependencies", []) or []:
            dep_task = ctx.plan.get(dep_id)
            if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    # -----------------------------------------------------------------------
    # Task execution
    # -----------------------------------------------------------------------

    @traced(capture_input=False)
    async def _execute_ready_tasks(
        self, tasks: Any, ctx: RuntimeState, plan_lock: asyncio.Lock
    ) -> list[TaskItem]:
        """Execute all tasks selected by the supervisor that are ready to run.

        Tasks whose dependencies are satisfied are executed concurrently using
        an ``asyncio.TaskGroup``. The returned list contains the updated
        ``TaskItem`` instances after execution.
        """
        # Cascade cancellations (delegated to Scheduler)
        cascade_fn = getattr(self, "_cascade_cancellations_cb", None)
        if cascade_fn:
            await cascade_fn(ctx)

        requested_ids: list[int] = list(dict.fromkeys(getattr(tasks, "tasks_to_execute", None) or []))

        candidate_steps: list[TaskItem] = [
            ctx.plan[task_id] for task_id in requested_ids if task_id in ctx.plan
        ]

        allowed_statuses = {TaskStatus.READY, TaskStatus.RERUN}

        ready_steps = [
            step
            for step in candidate_steps
            if step.status in allowed_statuses
            and self._dependencies_satisfied(step, ctx)
        ]
        if len(ready_steps) == 0:
            return []

        # Claim tasks (READY/RERUN -> RUNNING) atomically
        claimed_steps: list[TaskItem] = []
        async with plan_lock:
            for step in ready_steps:
                if step.status not in allowed_statuses:
                    continue
                if not self._dependencies_satisfied(step, ctx):
                    continue
                step.status = TaskStatus.RUNNING
                claimed_steps.append(step)
                await self._record_task_status_event_cb(
                    step.task_id,
                    TaskStatus.RUNNING,
                    reason="claimed_for_execution",
                )

        if not claimed_steps:
            return []

        # Prepare the concurrent coroutines
        ready_tasks = []
        for step in claimed_steps:
            if (
                getattr(tasks, "feedback_to_subagents", None)
                and step.task_id in tasks.feedback_to_subagents
            ):
                if step.parameters is None:
                    step.parameters = {}
                step.parameters["supervisor_feedback"] = (
                    tasks.feedback_to_subagents.get(step.task_id)
                )

            logger.info(
                f"- {step.task_id}: {step.sub_task_objective} using {step.capability}"
            )
            logger.info(f"  Dependencies: {step.sub_task_dependencies}")
            logger.info(f"  Status: {step.status}")
            logger.info(f"  Result: {step.result}")
            logger.info("\n")

            capability_registry = self._get_capability_registry()
            worker = capability_registry.get(step.capability)
            if worker:
                ready_tasks.append(self.run_task_cb(worker.tool_func, step, ctx))
            else:
                step.status = TaskStatus.ERRORED
                step.error_msg = f"Unknown capability: {step.capability!r}"
                await self._record_task_status_event_cb(
                    step.task_id,
                    TaskStatus.ERRORED,
                    reason="unknown capability",
                    error_msg=step.error_msg,
                )

        # Execute tasks
        logger.info("--- Executing Ready Tasks ---")
        task_results = []
        async with TaskGroup() as tg:
            for task in ready_tasks:
                task_results.append(tg.create_task(task))

        results = [t.result() for t in task_results]
        logger.info("--- All Ready Tasks Completed ---")
        return results

    @traced(run_type="task", capture_input=False)
    async def execute(
        self, capability: CapabilityRunner, step: TaskItem, runtime_state: RuntimeState
    ) -> TaskItem:
        """Execute a sub-agent for a single task and record the result.

        Builds a task-specific prompt (with optional supervisor feedback),
        runs the provided ``capability``, and updates the ``TaskItem`` status
        and result based on success or failure.
        """
        _feedback_for_agent = None
        if isinstance(step.parameters, dict):
            _feedback_for_agent = step.parameters.get("supervisor_feedback")

        if step.capability == "producer_agent":
            objective = self._get_objective()

            user_prompt = f"""
            Overall objective:
            {objective}

            You are the final synthesis agent.
            - First, call `list_completed_tasks` to see all completed upstream tasks.
            - For each task that is relevant to the objective (especially research tasks), call `get_task_result(task_id=...)`.
            - THEN, write a single, coherent comparative analysis answering the objective.
            - You MUST explicitly integrate evidence from ALL relevant completed tasks (e.g. Task 1 and Task 2 in this run).
            """

            if _feedback_for_agent:
                user_prompt += f"""

                    Supervisor feedback / additional instructions for this execution:

                    {_feedback_for_agent}
                    """

            user_prompt += """
                    Your job:
                    - Use ONLY the completed sub-task results from this run.
                    - Combine their findings into a single, coherent final answer.
                    - Follow your system prompt instructions for citations and final TaskResult structure.
                    - Do NOT request new research or create new sub-tasks.
                    """
        else:
            task_view: dict[str, Any] = step.model_dump(mode="json")
            params = task_view.pop("parameters", None)
            if isinstance(params, dict) and params:
                keys = sorted(list(params.keys()))
                task_view["parameters_keys"] = keys[:50]
                if len(keys) > 50:
                    task_view["parameters_keys_truncated"] = True

            task_json = json.dumps(task_view, indent=2, ensure_ascii=False)

            objective = self._get_objective()

            user_prompt = f"""
                You are executing TaskItem:

            {task_json}

                Overall objective:
                {objective}

                """
            if _feedback_for_agent:
                user_prompt += f"""

                Supervisor feedback / additional instructions for this execution:
                {_feedback_for_agent}
                """

            user_prompt += """

            ONLY act on this sub-task and any feedback. Do not re-plan or change the task.
            """
        task_deps = TaskRunDeps(runtime_state=runtime_state, task=step)

        user_prompt += f"""

Context-budget note:
- You may be running on a smaller-context model.
- Prefer small tool outputs. When calling tools that can return large text, request truncation.
- If you need a quick targeted answer from another capability, call `consult_capability(capability=..., question=...)`.
- Checkpoint progress frequently via `append_scratch_note(note=...)`.
"""

        max_resume_attempts = 2
        last_error: Exception | None = None

        for resume_attempt in range(max_resume_attempts + 1):
            tool_call_limit = 20 if resume_attempt == 0 else 10

            try:
                task_limits = self._make_usage_limits(
                    tool_calls_limit=tool_call_limit,
                    total_tokens_limit=self._remaining_token_budget(runtime_state),
                )
                result = await capability.run(
                    user_prompt,
                    deps=task_deps,
                    usage_limits=task_limits,
                )
                self._accumulate_usage(
                    runtime_state, result, label=f"task:{step.task_id}"
                )

                # Normalize to canonical TaskResult (required for critic/checkpointing).
                # Includes best-effort ingestion of file outputs into the artifact store.
                step.result = await self._coerce_output_cb(
                    step, result.output, runtime_state=runtime_state
                )

                # Merge any artifacts the agent attached via `attach_artifact_to_result`.
                if isinstance(step.result, TaskResult):
                    pending = step.metadata.get("result_artifacts")
                    if isinstance(pending, list) and pending:
                        existing_ids = {
                            a.artifact_id for a in (step.result.artifacts or [])
                        }
                        for item in pending:
                            try:
                                ar = ArtifactRef.model_validate(item)
                            except Exception:
                                continue
                            if ar.artifact_id in existing_ids:
                                continue
                            step.result.artifacts.append(ar)
                            existing_ids.add(ar.artifact_id)

                step.status = TaskStatus.NEEDS_REVIEW
                step.error_msg = None
                await self._record_task_result_cb(step)
                await self._record_task_status_event_cb(
                    step.task_id, TaskStatus.NEEDS_REVIEW
                )
                return step
            except Exception as e:
                last_error = e
                if (
                    self._is_context_limit_error(e)
                    and resume_attempt < max_resume_attempts
                ):
                    overflow_entry = {
                        "at": datetime.now().isoformat(),
                        "attempt": resume_attempt,
                        "error": str(e),
                    }
                    step.metadata.setdefault("context_overflow", [])
                    step.metadata["context_overflow"].append(overflow_entry)
                    await self._record_metadata_append_cb(
                        step.task_id, "context_overflow", overflow_entry
                    )

                    user_prompt = self._build_resume_prompt(step, e)
                    continue

                step.status = TaskStatus.ERRORED
                step.error_msg = str(e)
                await self._record_task_status_event_cb(
                    step.task_id,
                    TaskStatus.ERRORED,
                    error_msg=step.error_msg,
                )
                return step

        # Should be unreachable, but keep a safe fallback.
        step.status = TaskStatus.ERRORED
        step.error_msg = str(last_error) if last_error else "Unknown error"
        await self._record_task_status_event_cb(
            step.task_id, TaskStatus.ERRORED, error_msg=step.error_msg
        )
        return step
