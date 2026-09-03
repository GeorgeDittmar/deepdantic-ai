# Customization

You can add custom sub‑agents (capabilities) to your `DeepAgent` setup.

## Adding a Custom Capability

`DeepAgent` accepts additional capabilities via the `sub_agents` parameter. Each capability is described by a `CapabilityDescription` (see API reference) and can wrap either:

- a full `pydantic_ai.Agent` instance, or
- a simple async/sync callable (a plain tool function).

These are merged with the built‑in capabilities (`producer_agent`, `research_agent`,
`worker_agent`) inside `DeepAgent._setup_capabilities(...)`.

### Example: custom sub‑agent (Agent)

```python
from pydantask.agents.agent import DeepAgent
from pydantask.models import CapabilityDescription, TaskResult, TaskRunDeps
from pydantic_ai import Agent

my_sub_agent = Agent(
    model=...,  # e.g. an OpenAIChatModel
    name="_my_special_agent",
    system_prompt="You are a custom agent for specialized reasoning tasks.",
    deps_type=TaskRunDeps,   # gives tools access to deps.runtime_state + deps.task
    output_type=TaskResult,
    tools=[...],             # any tools this agent should be able to call
)

custom_description = CapabilityDescription(
    name="my_special_agent",  # used in TaskItem.capability
    description="Custom agent for specialized reasoning tasks.",
    tool_func=my_sub_agent,
)

agent = DeepAgent(objective="...", sub_agents=[custom_description])
```

### Example: simple callable as a capability (wrap with `as_runner`)

`DeepAgent` executes capabilities by calling a `.run(prompt, deps, usage_limits=...)` method.

- If you provide a `pydantic_ai.Agent`, it already has `.run(...)`.
- If you provide a plain function, wrap it with `as_runner(...)`.

```python
from pydantask.agents.agent import DeepAgent
from pydantask.capabilities.runner_v2 import as_runner
from pydantask.models import CapabilityDescription, TaskResult, TaskRunDeps


async def my_utility_capability(prompt: str, deps: TaskRunDeps) -> TaskResult:
    # deps.runtime_state is the shared run state; deps.task is the current TaskItem
    return TaskResult(
        task_id=deps.task.task_id,
        summary="Utility processed the prompt",
        detailed_output=f"Got prompt (len={len(prompt)}):\n\n{prompt}",
    )


my_utility_capability_desc = CapabilityDescription(
    name="my_utility_tool",
    description="Utility capability that transforms a prompt into a TaskResult.",
    tool_func=as_runner(my_utility_capability),
)

agent = DeepAgent(objective="...", sub_agents=[my_utility_capability_desc])
```

The planner sees the `name` and `description` in `CapabilityDescription` and may choose that capability when constructing `TaskItem.capability` values.

## Accessing shared state in Custom Agents

If your sub‑agent (a `pydantic_ai.Agent`) needs access to the shared plan/state, it should use `deps_type=TaskRunDeps` (this is what `DeepAgent` passes during task execution):

```python
from pydantask.models import TaskResult, TaskRunDeps
from pydantic_ai import Agent

my_context_agent = Agent(
    model=...,
    name="_my_context_agent",
    system_prompt="You can inspect the shared plan via deps.runtime_state and the current task via deps.task.",
    deps_type=TaskRunDeps,
    output_type=TaskResult,
    tools=[...],
)
```

In this case, when `DeepAgent` calls `my_context_agent.run(...)`, it passes `deps=TaskRunDeps(runtime_state=..., task=...)`, and your tools can access `ctx.deps.runtime_state.plan`, `ctx.deps.runtime_state.document_store`, etc.

**Important:**

- Capabilities that need to interact with DeepAgent's orchestration (plan, task statuses, shared documents) typically should:
  - use `deps_type=TaskRunDeps` on their `Agent` (so tools can access both the shared `runtime_state` and the current `task`), and
  - write tools that accept `RunContext[TaskRunDeps]`.

  If you truly need supervisor-style access (operate on the whole run outside task execution), use `deps_type=RuntimeState` and tools that accept `RunContext[RuntimeState]`.
- Agents with different context/state types are possible but integration with `DeepAgent`'s orchestration is not guaranteed; you must handle such cases yourself.
- Stateless agents or tools (that do not use shared runtime state) do not need to accept `deps` or `RunContext`.

For more API detail, see [API Reference](api.md).
