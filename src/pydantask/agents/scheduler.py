"""Scheduler — deterministic DAG state normalization.

Extracted from DeepAgent so the scheduler logic can be tested in isolation.

Design
------
The Scheduler class receives *only* the data it needs via a context_resolver
callable.  At call time it extracts:

* ``plan`` – the shared ``dict[int, TaskItem]``
* ``capability_registry`` – capability name → description mapping
* ``_plan_lock`` – asyncio.Lock for plan mutations
* ``_record_task_status_event`` – checkpoint callback

This keeps the module independent of DeepAgent's internal structure.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Optional

from loguru import logger
from pydantic_ai import RunContext

from pydantask.capabilities.introspection import (
    callable_input_schema,
    unwrap_callable,
)
from pydantask.models import (
    SupervisorDecision,
    TaskItem,
    TaskQAResult,
    TaskResult,
    TaskStatus,
)


class Scheduler:
    """Deterministic DAG state normalization for the DeepAgent control loop.

    This class encapsulates all non-LLM state transitions that keep the task
    DAG in a consistent state each cycle:

    - Unknown capability detection (marks tasks ERRORED)
    - Callable parameter self-heal (promotes errored tasks back to READY)
    - Missing parameter erroring (transitions tasks to ERRORED)
    - Dependency-based readiness propagation (PENDING ↔ READY)
    - Final result selection
    - Deadlock reporting
    - Seed plan application
    - Cancellation cascading

    Typical usage::

        scheduler = Scheduler(
            context_resolver=lambda ctx: {
                "plan": ctx.deps.plan,
                "capability_registry": ctx.deps.capability_registry,
                "plan_lock": ctx.deps._plan_lock,
                "record_task_status_event": ctx.deps._record_task_status_event,
            },
        )
        report = await scheduler.scheduler_pass(runtime_state)
    """

    def __init__(
        self,
        context_resolver: Callable[[Any], Any],
    ) -> None:
        """Initialize Scheduler.

        Args:
            context_resolver: Called with each ``RuntimeState`` to extract the
                values the scheduler methods need.
        """
        self._resolver = context_resolver

    @staticmethod
    def is_terminal_status(status: TaskStatus) -> bool:
        """Return True if a task is in a terminal state.

        Note: ERRORED is intentionally treated as non-terminal; the supervisor
        may still choose to patch the task and rerun it.
        """
        return status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

    @staticmethod
    def dependencies_satisfied(step: TaskItem, ctx: Any) -> bool:
        """Return ``True`` if all of a task's dependencies are fully satisfied.

        Currently a dependency is considered satisfied only if the dependent
        task exists and is in the ``COMPLETED`` state.
        """
        required_statuses = {TaskStatus.COMPLETED}
        for dep_id in step.sub_task_dependencies or []:
            dep_task = ctx.plan.get(dep_id)
            if dep_task is None:
                return False
            if dep_task.status not in required_statuses:
                return False
        return True

    async def scheduler_pass(self, runtime_state: Any) -> str:
        """Deterministic scheduler pass — non-LLM state normalization.

        Runs each control-loop cycle to keep the task DAG in a consistent state.
        Performs four categories of work:

        1. **Unknown capability detection** — marks tasks with unrecognized
           ``capability`` names as ERRORED (unless already in a terminal state).
        2. **Callable parameter self-heal** — for callable capabilities, verifies
           that all required function parameters are present in
           ``TaskItem.parameters``. If a task was previously errored for missing
           params and the supervisor has since patched them, promotes it back to
           READY (if deps are satisfied).
        3. **Missing parameter erroring** — if required parameters are still
           missing, transitions PENDING/READY/RERUN tasks to ERRORED with a
           message telling the supervisor what to patch.
        4. **Dependency-based readiness** — promotes PENDING → READY when all
           dependencies are COMPLETED; demotes READY → PENDING if deps are no
           longer satisfied.

        Args:
            runtime_state: The current ``RuntimeState`` containing the task plan.

        Returns:
            A human-readable report string describing all status changes.
            Returns ``"No scheduler changes this cycle."`` if nothing changed.
        """
        s = self._resolver(runtime_state)
        plan = s.plan
        lock = s.plan_lock
        registry = s.capability_registry
        record_event = s.record_task_status_event

        changes: list[str] = []
        warnings: list[str] = []

        async with lock:
            for task_id, task in sorted(plan.items(), key=lambda kv: kv[0]):
                # Unknown capability detection.
                if task.capability and task.capability not in registry:
                    if not Scheduler.is_terminal_status(task.status):
                        if task.status != TaskStatus.ERRORED:
                            changes.append(
                                f"- Task {task_id}: {task.status.value} -> errored (unknown capability: {task.capability!r})"
                            )
                            task.status = TaskStatus.ERRORED
                            task.error_msg = f"Unknown capability: {task.capability!r}"
                            await record_event(
                                task_id,
                                TaskStatus.ERRORED,
                                reason="unknown capability",
                                error_msg=task.error_msg,
                            )
                        else:
                            task.error_msg = f"Unknown capability: {task.capability!r}"
                    continue

                # Deterministic callable input contract check.
                cap_desc = registry.get(task.capability) if task.capability else None
                func = unwrap_callable(getattr(cap_desc, "tool_func", None)) if cap_desc else None
                if func is not None:
                    schema = callable_input_schema(func)
                    required = list(schema.get("required") or [])
                    if required:
                        params = getattr(task, "parameters", None)
                        if not isinstance(params, dict):
                            params = {}

                        missing = [k for k in required if k not in params]

                        # Self-heal: if the task was previously errored for missing parameters
                        # and the supervisor has since patched them in, promote back to runnable.
                        if (
                            not missing
                            and task.status == TaskStatus.ERRORED
                            and task.metadata.get("errored_reason")
                            == "missing_required_parameters"
                        ):
                            deps_ok_now = Scheduler.dependencies_satisfied(task, runtime_state)
                            new_status = (
                                TaskStatus.READY if deps_ok_now else TaskStatus.PENDING
                            )
                            changes.append(
                                f"- Task {task_id}: errored -> {new_status.value} (required parameters supplied)"
                            )
                            task.status = new_status
                            task.error_msg = None
                            task.metadata.pop("missing_parameters", None)
                            task.metadata.pop("errored_reason", None)
                            await record_event(
                                task_id,
                                new_status,
                                reason="required_parameters_supplied",
                            )

                        if missing and task.status in {
                            TaskStatus.PENDING,
                            TaskStatus.READY,
                            TaskStatus.RERUN,
                        }:
                            msg = (
                                "Missing required parameters for callable capability "
                                f"{task.capability!r}: missing={missing}; required={required}. "
                                "Supervisor must patch the task with parameters={...}."
                            )
                            changes.append(
                                f"- Task {task_id}: {task.status.value} -> errored (missing required parameters: {missing})"
                            )
                            task.status = TaskStatus.ERRORED
                            task.error_msg = msg
                            task.metadata["missing_parameters"] = missing
                            task.metadata["errored_reason"] = "missing_required_parameters"
                            await record_event(
                                task_id,
                                TaskStatus.ERRORED,
                                reason="missing_required_parameters",
                                error_msg=msg,
                            )
                            continue

                # Dependency-based readiness propagation.
                deps_ok = Scheduler.dependencies_satisfied(task, runtime_state)

                if task.status == TaskStatus.PENDING and deps_ok:
                    task.status = TaskStatus.READY
                    changes.append(
                        f"- Task {task_id}: pending -> ready (deps satisfied)"
                    )
                    await record_event(
                        task_id,
                        TaskStatus.READY,
                        reason="dependencies_satisfied",
                    )

                # Keep READY tasks honest if deps are not actually satisfied.
                if task.status == TaskStatus.READY and not deps_ok:
                    task.status = TaskStatus.PENDING
                    changes.append(
                        f"- Task {task_id}: ready -> pending (deps not satisfied)"
                    )
                    await record_event(
                        task_id,
                        TaskStatus.PENDING,
                        reason="dependencies_not_met",
                    )

        if not changes and not warnings:
            return "No scheduler changes this cycle."

        out: list[str] = []
        if changes:
            out.append("Status normalization:")
            out.extend(changes)
        if warnings:
            out.append("Warnings:")
            out.extend(warnings)
        return "\n".join(out)

    def select_final_result(self, runtime_state: Any) -> TaskResult | None:
        """Select the run's final output deterministically.

        Priority:
        1) A COMPLETED task with ``is_final=True``.
        2) A COMPLETED task with non-empty ``result.detailed_output``.
        3) A COMPLETED ``producer_agent`` task.
        4) Otherwise the newest COMPLETED task with any result.

        This ensures checkpoint resume returns a stable final deliverable even
        if the supervisor immediately declares completion.
        """
        completed: list[TaskItem] = [
            t
            for t in runtime_state.plan.values()
            if t.status == TaskStatus.COMPLETED and t.result is not None
        ]
        if not completed:
            return None

        finals = [t for t in completed if getattr(t, "is_final", False)]
        if finals:
            return max(finals, key=lambda t: t.task_id).result

        with_detail = [
            t
            for t in completed
            if (t.result and (t.result.detailed_output or "").strip())
        ]
        if with_detail:
            return max(with_detail, key=lambda t: t.task_id).result

        producers = [t for t in completed if t.capability == "producer_agent"]
        if producers:
            return max(producers, key=lambda t: t.task_id).result

        return max(completed, key=lambda t: t.task_id).result

    def build_deadlock_report(
        self, runtime_state: Any, decision: SupervisorDecision | None = None
    ) -> str:
        """Explain why no tasks ran in the current cycle."""
        s = self._resolver(runtime_state)
        plan = s.plan
        registry = s.capability_registry

        status_counts = Counter(t.status.value for t in plan.values())
        runnable: list[int] = []
        blocked: list[str] = []

        for task_id, task in sorted(plan.items(), key=lambda kv: kv[0]):
            if Scheduler.is_terminal_status(task.status):
                continue

            if task.status in {
                TaskStatus.READY,
                TaskStatus.RERUN,
            } and Scheduler.dependencies_satisfied(task, runtime_state):
                runnable.append(task_id)
                continue

            # Compute a human-readable reason.
            if task.capability not in registry:
                blocked.append(
                    f"- Task {task_id} [{task.status.value}]: unknown capability {task.capability!r}"
                )
                continue

            if task.sub_task_dependencies:
                missing = [d for d in task.sub_task_dependencies if d not in plan]
                if missing:
                    blocked.append(
                        f"- Task {task_id} [{task.status.value}]: missing deps {missing}"
                    )
                    continue

                unmet = [
                    d
                    for d in task.sub_task_dependencies
                    if plan.get(d) is not None
                    and plan[d].status != TaskStatus.COMPLETED
                ]
                if unmet:
                    blocked.append(
                        f"- Task {task_id} [{task.status.value}]: waiting on deps {unmet}"
                    )
                    continue

            blocked.append(f"- Task {task_id} [{task.status.value}]: not runnable")

        lines: list[str] = []
        lines.append("Deadlock / no-progress report:")
        lines.append(f"- status_counts: {dict(status_counts)}")
        if decision is not None:
            lines.append(f"- supervisor_requested: {decision.tasks_to_execute or []}")
        lines.append(f"- runnable_now: {runnable}")
        if blocked:
            lines.append("- blocked_examples:")
            lines.extend(blocked[:12])

        return "\n".join(lines)

    def apply_seed_plan(self, runtime_state: Any, seed_plan: Any) -> None:
        """Seed ``runtime_state.plan`` from a ``Plan`` if provided.

        This is used to support user-specified plans. It validates:

        * Unique task IDs.
        * Dependencies refer to existing tasks.

        It also updates ``runtime_state.next_task_id``.
        """
        if seed_plan is None:
            return

        tasks = list(seed_plan.tasks or [])
        if not tasks:
            return

        plan_dict: dict[int, TaskItem] = {}
        for t in tasks:
            if t.task_id in plan_dict:
                raise ValueError(f"Duplicate task_id in seed_plan: {t.task_id}")
            if not getattr(t, "overall_objective", None):
                t.overall_objective = runtime_state.objective
            plan_dict[t.task_id] = t

        for t in plan_dict.values():
            for dep_id in t.sub_task_dependencies or []:
                if dep_id not in plan_dict:
                    raise ValueError(
                        f"seed_plan task {t.task_id} depends on missing task {dep_id}"
                    )

        runtime_state.plan = plan_dict
        runtime_state.next_task_id = max(plan_dict.keys()) + 1

    async def cascade_cancellations(self, runtime_state: Any) -> None:
        """Transitively marks downstream tasks as CANCELLED if they rely
        on an upstream task that has been cancelled.
        """
        s = self._resolver(runtime_state)
        plan = s.plan
        lock = s.plan_lock
        record_event = s.record_task_status_event

        async with lock:
            changed = True
            while changed:
                changed = False
                for task in plan.values():
                    if task.status in {TaskStatus.PENDING, TaskStatus.READY}:
                        for dep_id in task.sub_task_dependencies or []:
                            dep_task = plan.get(dep_id)
                            if dep_task and dep_task.status == TaskStatus.CANCELLED:
                                task.status = TaskStatus.CANCELLED
                                task.error_msg = (
                                    f"Upstream dependency Task {dep_id} was cancelled."
                                )
                                await record_event(
                                    task.task_id,
                                    TaskStatus.CANCELLED,
                                    reason=f"Upstream task {dep_id} cancelled; dropping downstream branch.",
                                    error_msg=task.error_msg,
                                )
                                changed = True
                                break
