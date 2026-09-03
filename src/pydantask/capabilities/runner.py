from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Optional, Protocol, TypeVar

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from pydantask.models.models import TaskRunDeps  # adjust import path if different

T = TypeVar("T")

import asyncio
import inspect
from functools import partial

def is_async_callable(obj) -> bool:
    if not callable(obj):
        return False
    
    # Handle partial functions
    if isinstance(obj, partial):
        return is_async_callable(obj.func)
    
    # Check standard async functions, methods, and closures
    if inspect.iscoroutinefunction(obj):
        return True
    
    # Check classes with an asynchronous __call__ method
    if hasattr(obj, '__call__'):
        return inspect.iscoroutinefunction(obj.__call__)
        
    return False


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
    func: Callable[[str, TaskRunDeps], Awaitable[T]]

    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,  # kept for compatibility
    ) -> RunResult[T]:
        out = await self.func(prompt, deps)
        return RunResult(output=out)

@dataclass
class SyncFuncRunner(Generic[T]):
    func: Callable[[str, TaskRunDeps], T]

    async def run(
        self,
        prompt: str,
        *,
        deps: TaskRunDeps,
        usage_limits: Optional[UsageLimits] = None,
    ) -> RunResult[T]:
        out = await asyncio.to_thread(self.func, prompt, deps)
        return RunResult(output=out)

def as_runner(obj: Any) -> CapabilityRunner[Any]:
    """Centralized normalization (the only place you type-check)."""
    if isinstance(obj, Agent):
        return AgentRunner(obj)
    if callable(obj):
        # assume async callable(prompt, deps) by default
        print("RUNNING FUNCTIONS")
        return AsyncFuncRunner(obj)
    raise TypeError(f"Unsupported capability type: {type(obj)!r}")