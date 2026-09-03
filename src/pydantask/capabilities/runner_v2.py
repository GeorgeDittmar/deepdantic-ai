from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from pydantask.models.models import RuntimeState, TaskItem, TaskRunDeps

T = TypeVar("T")


def is_async_callable(obj: Any) -> bool:
    """Return True if ``obj`` should be awaited when invoked.

    This handles:
      - async functions
      - partial(async_fn, ...)
      - callable objects/classes with async ``__call__``
    """
    if not callable(obj):
        return False

    if isinstance(obj, partial):
        return is_async_callable(obj.func)

    if inspect.iscoroutinefunction(obj):
        return True

    call = getattr(obj, "__call__", None)
    if call is not None and inspect.iscoroutinefunction(call):
        return True

    return False


def _annotation_matches(annotation: Any, target_type: type) -> bool:
    if annotation is target_type:
        return True

    origin = get_origin(annotation)
    if origin is None:
        return False

    # Optional[T] is Union[T, NoneType]
    if origin is Union:
        return any(_annotation_matches(a, target_type) for a in get_args(annotation))

    return False


def _build_injected_call(
    func: Callable[..., Any],
    *,
    prompt: str,
    deps: TaskRunDeps,
    usage_limits: UsageLimits | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Build (args, kwargs) for calling an arbitrary capability function.

    Injection sources:
      - prompt (str)
      - deps (TaskRunDeps)
      - deps.runtime_state (RuntimeState)
      - deps.task (TaskItem)
      - usage_limits (UsageLimits | None)
      - deps.task.parameters (dict) as named args

    Rules (in order):
      1) If a parameter name is a known special name, inject it.
      2) Else, if the parameter name exists in task.parameters, inject that value.
      3) Else, if the annotation matches a known type, inject it.
      4) Else, if it has a default, omit it.
      5) Else, raise a TypeError (missing required input).

    If the function accepts **kwargs, any *unused* task.parameters are forwarded.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        # If we can't introspect, fall back to the legacy convention.
        return [prompt, deps], {}

    task_params: dict[str, Any] = {}
    if isinstance(getattr(deps.task, "parameters", None), dict):
        task_params = dict(deps.task.parameters)

    special_by_name: dict[str, Any] = {
        # prompt aliases
        "prompt": prompt,
        "text": prompt,
        "input": prompt,
        "question": prompt,
        "query": prompt,
        # deps/state/task
        "deps": deps,
        "task_deps": deps,
        "runtime_state": deps.runtime_state,
        "state": deps.runtime_state,
        "runtime": deps.runtime_state,
        "task": deps.task,
        "step": deps.task,
        # limits
        "usage_limits": usage_limits,
        "limits": usage_limits,
        # parameters dict passthrough
        "parameters": task_params,
        "params": task_params,
        "task_parameters": task_params,
    }

    ann_by_type: dict[type, Any] = {
        TaskRunDeps: deps,
        RuntimeState: deps.runtime_state,
        TaskItem: deps.task,
        UsageLimits: usage_limits,
    }

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    used_param_keys: set[str] = set()
    accepts_var_kw = False

    for name, p in sig.parameters.items():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_kw = True
            continue

        # If the function explicitly wants `*args`, we don't try to invent any.
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            continue

        value_set = False
        value: Any = None

        if name in special_by_name:
            value = special_by_name[name]
            value_set = True
        elif name in task_params:
            value = task_params[name]
            used_param_keys.add(name)
            value_set = True
        else:
            for ann_type, ann_value in ann_by_type.items():
                if _annotation_matches(p.annotation, ann_type):
                    value = ann_value
                    value_set = True
                    break

        if not value_set:
            if p.default is not inspect._empty:
                continue
            raise TypeError(
                "Cannot call capability function; missing required argument "
                f"{name!r}. Provide it via TaskItem.parameters or use a recognized name "
                f"(e.g. 'prompt', 'deps', 'task', 'runtime_state'). Available parameter keys: "
                f"{sorted(task_params.keys())}"
            )

        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value

    if accepts_var_kw:
        for k, v in task_params.items():
            if k not in used_param_keys and k not in kwargs:
                kwargs[k] = v

    return args, kwargs


@dataclass
class RunResult(Generic[T]):
    output: T


class CapabilityRunner(Protocol[T]):
    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,
    ) -> RunResult[T]: ...


@dataclass
class AgentRunner(Generic[T]):
    agent: Agent

    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,
    ) -> RunResult[T]:
        r = await self.agent.run(prompt, deps=deps, usage_limits=usage_limits)
        return RunResult(output=r.output)


@dataclass
class AsyncFuncRunner(Generic[T]):
    """Run an *async* callable capability with signature injection."""

    func: Callable[..., Awaitable[T]]

    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,  # kept for compatibility
    ) -> RunResult[T]:
        args, kwargs = _build_injected_call(
            self.func, prompt=prompt, deps=deps, usage_limits=usage_limits
        )
        out = await self.func(*args, **kwargs)
        return RunResult(output=out)


@dataclass
class SyncFuncRunner(Generic[T]):
    """Run a *sync* callable capability in a thread with signature injection."""

    func: Callable[..., T]

    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,
    ) -> RunResult[T]:
        args, kwargs = _build_injected_call(
            self.func, prompt=prompt, deps=deps, usage_limits=usage_limits
        )
        out = await asyncio.to_thread(self.func, *args, **kwargs)
        return RunResult(output=out)


def as_runner(obj: Any) -> CapabilityRunner[Any]:
    """Normalize Agents and arbitrary callables into a ``CapabilityRunner``.

    Supported inputs:
      - pydantic_ai.Agent
      - any async or sync callable
      - an existing runner object with an async ``.run(...)`` method

    For callables, the wrapper uses the target function's signature to inject
    arguments (prompt/deps/task/runtime_state/usage_limits and TaskItem.parameters).
    """
    if isinstance(obj, Agent):
        return AgentRunner(obj)

    # Already a runner-like object.
    run_attr = getattr(obj, "run", None)
    if run_attr is not None and callable(run_attr) and not callable(obj):
        return obj  # type: ignore[return-value]

    if callable(obj):
        return AsyncFuncRunner(obj) if is_async_callable(obj) else SyncFuncRunner(obj)

    raise TypeError(f"Unsupported capability type: {type(obj)!r}")
