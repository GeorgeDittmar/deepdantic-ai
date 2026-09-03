# Pydantask — Codebase Wiki

> Alpha harness for building deep, multi-step agents on top of [Pydantic AI](https://ai.pydantic.dev/).
> Version: 0.1.0a6 | Python >=3.12 | License: Apache-2.0

## Table of Contents

- [Project Structure](#project-structure)
- [High-Level Architecture](#high-level-architecture)
- [Entry Point & Quickstart](#entry-point--quickstart)
- [Core Data Models](#core-data-models)
- [The DeepAgent Orchestrator](#the-deepagent-orchestrator)
- [Control Loop](#control-loop)
- [Sub-Agents](#sub-agents)
- [Capability System](#capability-system)
- [Scheduler](#scheduler)
- [Tools](#tools)
- [Checkpointing & Resume](#checkpointing--resume)
- [Tracing & Observability](#tracing--observability)
- [Memory Layer](#memory-layer)
- [Workflow YAML](#workflow-yaml)
- [Clean Code & Design Practices](#clean-code--design-practices)
- [Development Conventions](#development-conventions)

---

## Project Structure

```
src/
├── pydantask/
│   ├── agents/
│   │   ├── agent.py          # DeepAgent (~2700 lines, core orchestrator)
│   │   ├── spec.py           # AgentSpec classes (SupervisorSpec, ResearcherSpec, etc.)
│   │   ├── utils.py          # Spec utilities
│   │   └── factory.py        # Agent factory (minimal)
│   ├── capabilities/
│   │   ├── runner_v2.py      # CapabilityRunner protocol + signature injection
│   │   ├── runner.py         # Legacy runner (empty stub)
│   │   └── introspection.py  # callable_input_schema, unwrap_callable
│   ├── capability/           # Legacy (minimal)
│   ├── capability.py         # Legacy alias
│   ├── manager/
│   │   ├── TaskManager.py    # TaskManager (minimal stub)
│   │   └── checkpointer.py   # CheckpointEvent, CheckpointRecorder
│   ├── memory/
│   │   ├── core.py           # MemoryLayer (SQLite-backed)
│   │   ├── schemas.py        # MemoryItem, Fact, SCHEMA_SQL
│   │   ├── embedder.py       # Hash/transformers embedding
│   │   └── tools.py          # ToolDef registry for memory tools
│   ├── models/
│   │   └── models.py         # All core data models (~730 lines)
│   ├── observe/
│   │   └── tracing.py        # Multi-backend tracing
│   ├── prompts/
│   │   ├── prompts.py        # Legacy prompts
│   │   └── prompts_v2.py     # Current prompts (~900 lines)
│   ├── skills/               # Skill stubs (empty)
│   ├── specs/                # Spec stubs (empty)
│   ├── tools/
│   │   ├── default_tools.py  # Core tools (think, scratch, file, fetch)
│   │   ├── artifact_tools.py # Content-addressed artifact store
│   │   └── memory_tools.py   # Knowledge store tools
│   ├── workflow/
│   │   └── yaml.py           # YAML workflow import & DAG validation
│   └── __init__.py
├── example.py                # Example usage
└── tmp/                      # Temp files
test/
├── test_agent.py             # Agent tests
├── test_default_tools_filesystem.py
├── test_fetch_url_content_tool.py
└── test_workflow_yaml.py
docs/                         # MkDocs documentation
examples/                     # Example projects
```

---

## High-Level Architecture

Pydantask is a **supervisor-driven multi-agent orchestrator** that manages a dynamic task DAG (directed acyclic graph). A `DeepAgent` instance coordinates four built-in sub-agents over shared mutable state:

```
┌─────────────────────────────────────────────────────┐
│                    DeepAgent                         │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ Supervisor│───▶│ Researcher│    │ Producer │       │
│  │  (DAG    │    │ (Web     │    │ (Synthes │       │
│  │  Architect)│   │  Search) │    │  iser)   │       │
│  └──────────┘    └──────────┘    └──────────┘       │
│       ▲                  │                  │        │
│       └───────┐          ▼                  │        │
│               ▼                               │        │
│         ┌──────────┐                           │        │
│         │  Critic  │◀────── (evaluates every  │        │
│         │  (QA)    │        worker output)    │        │
│         └──────────┘                           │        │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  RuntimeState       │  ← shared mutable context
│  - plan             │    (dict[int, TaskItem])
│  - objective        │
│  - capability_reg   │
│  - document_store   │
│  - knowledge_store  │
└─────────────────────┘
```

The supervisor incrementally builds/patches the DAG. Ready tasks execute in parallel via `asyncio.TaskGroup`. Each task result flows through a critic for QA before the supervisor decides the next cycle.

## Entry Point & Quickstart

```python
from pydantask.agents import DeepAgent
import asyncio

async def main():
    agent = DeepAgent(
        objective="Your goal here",
        model="openai:gpt-4.1-mini",  # or "anthropic:..." or Model instance
        max_steps=10,
        trace=False,
        checkpoint=False,
    )
    result = await agent.run()
    print(result.final_result.detailed_output if result.final_result else result.errors)

asyncio.run(main())
```

Environment:
- `OPENAI_API_KEY` — required
- `TAVILY_API_KEY` — optional; if absent the researcher falls back to DuckDuckGo

## Core Data Models

All in `src/pydantask/models/models.py`.

| Model | Purpose |
|---|---|
| `Plan` | Planner output: `reasoning_steps` + `list[TaskItem]`. Converted into `RuntimeState.plan`. |
| `TaskItem` | One sub-task in the DAG. Key fields: `task_id`, `objective`, `capability`, `dependencies`, `status`, `result`, `is_final`, `attempt_count`/`max_attempts`. |
| `TaskResult` | Canonical output from any worker. Fields: `task_id`, `status`, `summary`, `detailed_output`, `sources`, `artifacts`, `data`, `error_msg`, `metadata`. |
| `TaskQAResult` | Critic evaluation: `task_id`, `reasoning`, `passed`, `task_feedback`. |
| `TaskStatus` | Enum: `PENDING` → `READY` → `RUNNING` → `NEEDS_REVIEW` → `COMPLETED` / `RERUN` / `FAILED` / `ERRORED` / `CANCELLED`. |
| `RuntimeState` | Shared mutable state passed between agents. Holds `plan`, `objective`, `capability_registry`, `document_store`, `knowledge_store`, `runtime_steps`, `tokens_used`. |
| `SupervisorDecision` | Output of the supervisor: `reasoning`, `tasks_to_execute`, `feedback_to_subagents`, `all_tasks_completed`. |
| `CapabilityDescription` | Metadata + implementation for a sub-agent/tool. Fields: `name`, `description`, `tool_func`, `input_schema`. |
| `PydanTaskRunResult` | Return type of `DeepAgent.run()`: wraps `final_result`, `plan`, `runtime_state`, `errors`. |
| `SourceRef` / `ArtifactRef` / `KnowledgeRecord` | Structured references for citations, stored files, and knowledge artifacts. |
| `WorkflowYamlConfig` / `WorkflowTaskConfig` | Strict user-facing YAML schema for pre-defined DAGs. |

## The DeepAgent Orchestrator

`src/pydantask/agents/agent.py` (~2700 lines). This is the single most important file.

### Constructor (`__init__`)

Initializes:
1. **Model** — resolves provider-prefixed strings (`"openai:..."`, `"anthropic:..."`) or accepts a `pydantic_ai.Model` instance. Wraps the HTTP client with `AsyncTenacityTransport` for retry on rate limits/transient errors.
2. **Supervisor agent** (`_supervisor_agent`) — `pydantic_ai.Agent` with `output_type=SupervisorDecision`. Tools: `add_task`, `patch_task`, `cancel_task`, `mark_final_task`, `update_task_status`, `view_qa_report`, `think_tool`, `get_current_datetime`.
3. **Researcher agent** (`_researcher_agent`) — web research with Tavily/DuckDuckGo + scratch notes + artifacts.
4. **Producer agent** (`producer_agent`) — final synthesis. No web search; relies on `list_completed_tasks` and `get_task_result`.
5. **Critic agent** (`_critic_agent`) — QA evaluator with `output_type=TaskQAResult`.
6. **Capability registry** — built-in capabilities (`research_agent`, `worker_agent`, `producer_agent`) + user-supplied ones.
7. **Checkpoint recorder** — if `checkpoint=True`, initializes `CheckpointRecorder` writing to `_checkpoint/<uuid>/`.

### Key Methods

| Method | Role |
|---|---|
| `run()` | Main control loop (see [Control Loop](#control-loop)). |
| `execute()` | Runs a single task's capability, coerces output to `TaskResult`, handles context overflow retries. |
| `_execute_ready_tasks()` | Claims READY tasks under `_plan_lock`, executes them in parallel via `TaskGroup`. |
| `add_task()` / `patch_task()` / `cancel_task()` | Supervisor tools for DAG mutation. |
| `mark_final_task()` | Sets exactly one task as `is_final=True`. |
| `_scheduler_pass()` | Deterministic readiness propagation (PENDING→READY when deps satisfied, self-heals errored tasks). |
| `_select_final_result()` | Deterministic final output selection (priority: `is_final=True` → has detail → producer_agent → newest). |
| `consult_capability()` | Agent-to-agent bounded consult tool (max 1200 tokens, no tool calls). |
| `handle_critic_result()` | Applies critic QA: `passed` → COMPLETED, `failed` + retries → RERUN (with feedback appended to objective), `failed` + exhausted → FAILED. |
| `_cascade_cancellations()` | Transitively cancels downstream tasks when an upstream is CANCELLED. |

## Control Loop

The main loop in `run()` runs for at most `max_steps` iterations:

1. **Token budget check** — abort if global budget exceeded.
2. **Scheduler pass** — `_scheduler_pass()` normalizes readiness and self-heals.
3. **Supervisor call** — formats the prompt (objective + status board + capabilities) and calls the supervisor agent. On first step, injects `BOOTSTRAP_INSTURCT` (planning mode); otherwise `ORCHESTRATION_INSTRUCT`.
4. **Completion check** — validates invariants: exactly one `is_final=True` task and it must be COMPLETED with a result. Overrides if invariants fail.
5. **Execute tasks** — `_execute_ready_tasks()` runs supervisor-selected READY tasks in parallel.
6. **Critic evaluation** — for each result, runs the critic agent which returns `TaskQAResult`. `handle_critic_result()` applies the verdict.
7. **Checkpoint** — if enabled, persists state summary.
8. **No-progress guardrail** — 3 consecutive cycles with no executed tasks → abort with deadlock report.

## Sub-Agents

All are `pydantic_ai.Agent` instances sharing the same model.

### Supervisor (DAG Architect)

- **Role**: Plans the initial DAG, decides which tasks to run next, patches/cancels tasks, declares completion.
- **System prompt**: `COMPRESSED_SUPER_PROMPT` / `DYNAMIC_SUPERVISOR_SYS_PROMPT` (prompts_v2.py).
- **Key tools**: `add_task`, `patch_task`, `cancel_task`, `mark_final_task`, `update_task_status`, `view_qa_report`.
- **Repair protocol**: 3 levels — Patch (refine objective), Pivot (cancel + reroute), Replan (structural reset).

### Researcher

- **Role**: Web research with structured citation.
- **Tools**: Tavily/DuckDuckGo search, `think_tool`, `append_scratch_note`, `get_current_datetime`, artifacts.
- **System prompt**: `COMPRESSED_RESEARCH_SYS_PROMPT`.
- **Output**: `TaskResult` with `sources` populated as `SourceRef` objects.

### Producer

- **Role**: Final synthesis from completed task outputs.
- **Constraint**: No web search. Must rely solely on `list_completed_tasks` and `get_task_result`.
- **System prompt**: `COMPRESSED_PRODUCER_SYS_PROMPT`.
- **Critical**: Cannot request more information; must set status=errored if insufficient.

### Critic (QA Evaluator)

- **Role**: Evaluates each worker output against the sub-task objective.
- **Output**: `TaskQAResult` with `passed`, `reasoning`, `task_feedback`.
- **System prompt**: `COMPRESSED_CRITIC_SYS_PROMPT` / `CRITIC_SYS_PROMPT`.
- **Decision flow**: passed → `COMPLETED`; failed + retries < max → `RERUN` (feedback appended to objective); failed + max → `FAILED`.

## Capability System

The capability system lets users register arbitrary sub-agents or functions as DAG nodes.

### CapabilityDescription

```python
CapabilityDescription(
    name="my_agent",           # Used in TaskItem.capability
    description="What it does",
    tool_func=<agent or callable>,  # pydantic_ai.Agent or wrapped function
    input_schema=None,         # Optional Pydantic model class for structured input
)
```

### Runner Wrappers (`capabilities/runner_v2.py`)

`as_runner(obj)` normalizes any input into a `CapabilityRunner[T]` protocol:

| Input type | Runner | Behavior |
|---|---|---|
| `pydantic_ai.Agent` | `AgentRunner` | Calls `agent.run(prompt, deps, usage_limits)`. |
| Async callable | `AsyncFuncRunner` | Inspects signature, injects `prompt`/`deps`/`task`/`runtime_state`/`usage_limits` + `TaskItem.parameters`. |
| Sync callable | `SyncFuncRunner` | Same injection, runs via `asyncio.to_thread`. |
| Already a runner | Pass-through | Objects with a `.run(...)` method. |

### Signature Injection

For callable capabilities, `_build_injected_call()` maps function parameters to runtime values:

1. Named injection: `prompt`, `text`, `input`, `question`, `query` → task prompt string.
2. Context injection: `deps`, `task_deps`, `runtime_state`, `state`, `runtime` → `RuntimeState`.
3. Task injection: `task`, `step` → `TaskItem`.
4. Limits injection: `usage_limits`, `limits` → `UsageLimits`.
5. Parameters passthrough: `parameters`, `params`, `task_parameters` → `TaskItem.parameters` dict.
6. Annotation-based: matches type hints (`TaskRunDeps`, `RuntimeState`, `TaskItem`, `UsageLimits`).
7. **kwargs: any remaining `TaskItem.parameters` keys are forwarded.

### Callable Input Schema (`capabilities/introspection.py`)

`callable_input_schema(func)` introspects a function's signature to produce a JSON-schema-like dict showing required/optional parameters. Used to inform the supervisor about what structured inputs a callable capability needs.

## Scheduler

`_scheduler_pass()` is a deterministic, non-LLM pass that runs each cycle:

1. **Unknown capability detection** — marks tasks with unrecognized `capability` as ERRORED.
2. **Missing parameter self-heal** — for callable capabilities, checks `TaskItem.parameters` for required args. Errors if missing; self-heals back to READY once supervisor patches them.
3. **Dependency propagation** — PENDING → READY when all deps are COMPLETED; READY → PENDING if deps are no longer met.
4. **Report** — returns a human-readable report injected into the next supervisor prompt.

## Tools

### Default Tools (`tools/default_tools.py`)

| Tool | Purpose |
|---|---|
| `think_tool` | Private scratchpad for agent reasoning (returns empty string, no side effects). |
| `append_scratch_note` / `read_scratch_notes` | In-memory scratchpad per task, checkpointed. |
| `get_current_datetime` | Returns ISO-8601 current time. |
| `write_to_file_system` / `read_from_file_system` / `delete_from_file_system` | Filesystem tools (NOT enabled by default — harness is in-memory focused). |
| `list_documents` | Lists documents in `RuntimeState.document_store`. |
| `list_completed_tasks` / `get_task_result` | Cross-agent task result inspection. |
| `save_task_context` / `read_task_context` | Task-scoped file storage with convention `task-{id}-{kind}.md`. |
| `fetch_url_content` | Lightweight URL fetcher with SSRF protection, content-type gating, byte cap, truncation. |

### Artifact Tools (`tools/artifact_tools.py`)

Content-addressed file store (sha256) for durable, resumable artifact storage:

| Tool | Purpose |
|---|---|
| `put_artifact` | Store content; returns `ArtifactRef` JSON (id + uri). |
| `get_artifact` | Read artifact by id or uri (bounded to `max_chars`). |
| `list_artifacts` | List artifacts recorded on a task's metadata. |
| `attach_artifact_to_result` | Stage an `ArtifactRef` in `task.metadata["result_artifacts"]` for automatic merge into `TaskResult.artifacts`. |
| `store_file_as_artifact` | Internal: ingest a host-side file into the artifact store. |

Artifact root: checkpoint directory's `artifacts/` subfolder when checkpointing is enabled; otherwise falls back to `tools/tmp_files/artifacts/`.

### Memory Tools (`tools/memory_tools.py`)

Thin wrappers around `MemoryLayer` for `list_knowledge` / `get_knowledge` on the `RuntimeState.knowledge_store`.

## Checkpointing & Resume

Event-sourced durability via append-only JSONL log:

- **CheckpointRecorder** (`manager/checkpointer.py`): Writes `events.jsonl` and `summaries.jsonl`.
- **Event types**: `task_added`, `task_patched`, `task_status_updated`, `task_result`, `task_metadata_appended`, `scratch_note_appended`, `supervisor_decision`, `critic_feedback`, `final_task_set`.
- **Large result sidecars**: When `detailed_output` exceeds 4000 chars, full payload is saved to `{checkpoint}/task_results/task_{id}.json` and a truncated preview is stored in the event log.
- **Resume**: `_replay_checkpoint()` replays all events to reconstruct `RuntimeState`. `_apply_event()` is a typed dispatcher for each event type.
- **Task result persistence**: `_record_task_result()` stores the full `TaskResult` as JSON; if large, persists a sidecar and stores a pointer in the event.

## Tracing & Observability

Multi-backend tracing (`observe/tracing.py`):

| Backend | Env vars | Decorator |
|---|---|---|
| Langfuse | `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | `@observe(name=...)` |
| Logfire | `LOGFIRE_API_KEY` | `with logfire.span(name)` |
| LangSmith | `LANGCHAIN_API_KEY` / `LANGSMITH_API_KEY` | `@traceable(name=...)` |
| None | (default) | No-op |

`autodetect_tracing_backend()` picks the first available. `@traced()` decorator routes to the active backend. `Agent.instrument_all()` is called for Langfuse to show nested agent/model/tool spans.

## Memory Layer

`memory/` module provides a pluggable long-term memory system for agents:

### MemoryLayer (`memory/core.py`)

SQLite-backed with three memory types:
- **Episodic**: Events, experiences
- **Semantic**: Facts, knowledge
- **Procedural**: How-to's, processes

Key methods: `store()`, `search()` (semantic similarity), `get_recent()`, `store_fact()`, `get_facts()`, `build_context()`, `delete()`.

### Embedder (`memory/embedder.py`)

Falls back from `sentence-transformers` (MiniLM-L6-v2, 384-dim) to a deterministic hash-based bag-of-words embedding (256-dim, MD5-based) when no real model is installed.

### Memory Tools (`memory/tools.py`)

8 tool definitions: `memory_store`, `memory_search`, `memory_recent`, `memory_fact_set`, `memory_fact_get`, `memory_context`, `memory_delete`, `memory_stats`.

## Workflow YAML

`workflow/yaml.py` provides `import_yaml_workflow(path)` to load pre-defined DAGs from YAML files.

- Validates against `WorkflowYamlConfig` (strict schema, no unknown keys).
- Runs DAG validation: unique IDs, no self-dependencies, cycle detection via DFS.
- Auto-marks the highest task_id as `is_final=True` if not set.
- Converts to canonical `Plan` for use as `seed_plan` in `DeepAgent`.

---

## Clean Code & Design Practices

> **MANDATORY:** Any agent — LLM agent, sub-agent, or AI coding assistant — contributing to or modifying this codebase MUST adhere to the standards below. These are non-negotiable defaults. If you are unsure, ask.

### Guiding Principles

| Principle | Practice |
|---|---|
| **SOLID** | Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion. Every class, function, and module should have one reason to change. |
| **DRY** | No copied logic. Extract repeated patterns into named, tested functions. If you paste something three times, it belongs in a shared location. |
| **KISS** | Prefer the simplest solution that works. Avoid abstractions until repetition proves they are necessary. Complexity must be earned. |
| **YAGNI** | Do not build for hypothetical future needs. Implement only what is required by the current specification or failing test. |
| **Fail Fast** | Validate inputs at boundaries. Raise clear, typed errors early — never silently swallow exceptions or return `None` when an error is more appropriate. |
| **Immutability First** | Prefer immutable data structures and pure functions. Mutable state should be explicit, localized, and documented. |

### Naming & Readability

- **Variables**: `snake_case`, descriptive, noun-phrases. `user_count`, not `uc`. `is_valid` not `ok`.
- **Functions**: `snake_case`, verb-phrases. `validate_input`, `fetch_user_profile`.
- **Classes**: `PascalCase`, noun-phrases. `CheckpointRecorder`, `CapabilityRunner`.
- **Constants**: `UPPER_SNAKE_CASE`. `DEFAULT_TIMEOUT`, `MAX_RETRIES`.
- **Private**: Leading underscore — `_internal_method()`. Protected: `__mangled__` only when intentional.
- **Type hints on every function signature**. Even `() -> None`. The reader should not need to read the body to understand inputs/outputs.
- **Comments explain *why*, not *what***. Never comment obvious code — refactor it instead. If a comment explains what the code does, the code should be self-evident.

### Function Design

- **Max 4 levels of nesting**. If you are deeper, extract a function.
- **Max 10 parameters**. If you exceed, pack them into a dataclass or config object.
- **Single exit point** is a guideline, not a rule. Early returns are preferred over flag variables.
- **Functions should fit on one screen** (~30-40 lines). Longer = extract.
- **Prefer composition over inheritance**. Use protocols (`Protocol[T]`) for interfaces, not base classes.

### Module & File Design

- **Single responsibility per file**. One logical concept per file.
- **Max 300 lines per file** (excluding auto-generated code, type stubs, and test files).
- **Imports ordered**: stdlib → third-party → local. Grouped with blank lines between sections.
- **No circular imports**. If two modules import each other, extract the shared types into a third module.
- **`__init__.py` files are public API surface**, not implementation details. Export only what consumers need.

### Error Handling

- **Specific over generic**. `ValueError("task_id must be positive")` not `ValueError("bad input")`.
- **Never `except Exception:` without re-raising or logging**. Catch the narrowest exception possible.
- **Use Pydantic `Field(description=...)` for all model fields** so errors and docs are self-documenting.
- **Log at appropriate levels**: `logger.info()` for expected events, `logger.warning()` for recoverable issues, `logger.error()` for failures. Never `print()`.
- **Context matters in errors**: include enough information to diagnose without leaking secrets.

### Type Safety

- **`from __future__ import annotations`** at the top of every new file.
- **No `Any` unless unavoidable**. If used, add a comment explaining why and for how long.
- **Use `Optional[T]` (or `T | None`) explicitly**. Never rely on `None` defaults as a signal of optionality without the type hint.
- **Prefer `Literal` over string magic**: `status: Literal["pending", "running", "done"]` not `status: str`.
- **Use `Protocol` for structural subtyping** (duck typing with type safety) rather than deep class hierarchies.

### Testing

| Rule | Detail |
|---|---|
| **Tests are first-class code** | Same quality standards as production code: typed, documented, readable. |
| **One assert per test** (mostly) | Each test verifies one behavior. Multiple asserts are fine when they test a single atomic outcome. |
| **Test names are sentences** | `async def test_execute_returns_task_result_on_success()` — read like a spec. |
| **Arrange / Act / Assert** | Structure every test in three clearly separated blocks. |
| **Mock at boundaries** | Mock I/O (HTTP, filesystem, LLM calls), not pure logic. Test the real function, fake the outside world. |
| **No network in unit tests** | All external calls must be patched. Use `pytest-asyncio` for async tests. |
| **Cover the happy path AND the failure modes** | For every public function: what happens with valid input, invalid input, edge cases, and errors. |
| **Test files mirror source** | `src/pydantask/agents/agent.py` → `test/test_agent.py`. |
| **Run `pytest` before every commit** | No untested changes land in the repo. |

```bash
# Standard test command
pytest

# With coverage
pytest --cov=src/pydantask --cov-report=term-missing

# Specific test file
pytest test/test_agent.py -v

# Async tests
pytest test/test_agent.py -v --asyncio-mode=auto
```

### Documentation

| Rule | Detail |
|---|---|
| **Docstrings on every public function/class** | Google or NumPy style. At minimum: one-line summary, Args, Returns. |
| **README.md is the on-ramp** | New users should understand what the project is, how to install, and how to run their first example within 3 minutes. |
| **CLAUDE.md is the developer wiki** | Any AI agent or human developer should be able to understand and navigate the codebase from this file alone. |
| **CHANGELOG or git history tells the story** | Write meaningful commit messages: `fix: handle missing parameters in callable capabilities` not `fix stuff`. |
| **Inline comments for non-obvious *why*** | If the code is a workaround, has a known limitation, or depends on external behavior — document it. |

#### Docstring Template

```python
async def my_function(
    param: str,
    optional_param: int = 42,
) -> dict[str, Any]:
    """One-line summary of what this does.

    Longer description if needed. Explains the *why*, not the *what*.

    Args:
        param: Description of param.
        optional_param: Description with default.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param is empty.
        RuntimeError: When connection fails.
    """
```

### API & Interface Design

- **Explicit over implicit**. Prefer required parameters with defaults over magic behavior.
- **Backwards compatibility is a feature**. Deprecate gently with `warnings.warn()` before removing.
- **Version your public API**. Semantic versioning (semver) — MAJOR.MINOR.PATCH. Breaking changes → MAJOR.
- **Configuration over code for tunables**. Token limits, timeouts, retry counts — all configurable, not hardcoded.
- **Return rich error types** or at minimum structured dicts, not raw strings.

### Performance

- **Measure before optimizing**. Profiling comes first; optimization follows data.
- **I/O is the bottleneck**. Use async I/O (`asyncio`, `aiofiles`) for network and filesystem operations.
- **Truncate early, not late**. Limit data at the source (DB queries, tool outputs, LLM prompts) rather than filtering downstream.
- **State the complexity**. If an algorithm is O(n²) or worse, add a comment explaining why it is acceptable.

### Security

- **Never log secrets**. API keys, tokens, credentials — sanitize before logging.
- **Validate all external input**. User input, file paths, URLs — validate at the boundary before use.
- **SSRF protection** on any network tool. Block localhost, private IP ranges, DNS rebinding targets.
- **Least privilege**. Agents and tools should only have access to what they need. Default to read-only; write only when necessary.
- **Content-addressed storage** for artifacts (sha256). Prevents path traversal and ensures integrity.

### Agent-Specific Requirements

> Any agent — whether a `pydantic_ai.Agent` sub-agent, an AI coding assistant, or an autonomous agent contributing code — MUST follow these rules:

1. **No silent failures**. Every agent output path must handle errors explicitly. Never return `""` or `None` as a success signal.
2. **Structured outputs**. All agent outputs conform to Pydantic models (`TaskResult`, `SupervisorDecision`, etc.). No raw text where a schema exists.
3. **Tool contracts are contracts**. If a tool declares a parameter, it must be documented, type-hinted, and validated.
4. **Context budget awareness**. Agents should never dump entire datasets into LLM prompts. Use `max_chars`, pagination, and artifact references.
5. **Checkpoint progress**. Long-running agents call `append_scratch_note()` to save state. Recovery without checkpointing is unacceptable.
6. **No unbounded loops**. Every agent has a tool call limit (`tool_calls_limit`). Agents should reflect via `think_tool` before making more than 3 tool calls.
7. **Citation integrity**. Research agents must never invent sources. Every claim that needs backing requires a `SourceRef`.
8. **Deterministic fallbacks**. If an LLM call fails, the agent should fall back to cached state, scratch notes, or a structured error — not crash the run.

### Code Review Checklist

Before any PR (human or agent-generated):

- [ ] All new code is type-hinted
- [ ] All new public functions have docstrings
- [ ] Tests cover happy path, edge cases, and error paths
- [ ] No `Any` without a documented justification
- [ ] No hardcoded secrets, keys, or credentials
- [ ] Error messages are specific and actionable
- [ ] No `print()` — use `logger` instead
- [ ] No circular imports
- [ ] `pytest` passes with no warnings
- [ ] Changes are reflected in CLAUDE.md or docstrings where the public API changed
- [ ] Agent prompts are consistent with existing style (`COMPRESSED_*` for production)

---

## Development Conventions

### Adding a new sub-agent capability

1. Create a `pydantic_ai.Agent` with appropriate `system_prompt`, `tools`, `deps_type=TaskRunDeps`, `output_type=TaskResult`.
2. Wrap with `CapabilityDescription(name="...", description="...", tool_func=...)`.
3. If it's a plain callable (not an Agent), wrap with `as_runner(func)`.
4. Pass to `DeepAgent(capabilities=[...])`.

### Adding a new tool

1. Define an `async def tool(ctx: RunContext[RuntimeState], ...)` or `async def tool(prompt: str, deps: TaskRunDeps, ...)` function.
2. Add to the relevant agent's `tools=[]` list.
3. If it writes files, use `put_artifact`/`get_artifact` for content-addressed storage rather than raw paths.

### Prompt engineering

- Prefer `COMPRESSED_*` variants for production (shorter context, fewer tokens).
- Full variants (`*_SYS_PROMPT`) exist for readability during development.
- Supervisor prompts use `SUPERVISOR_INPUT_PROMPT.format(...)` with dynamic `plan_display` and `agent_display`.

### Concurrency model

- `_plan_lock` protects all plan mutations (task claiming, status updates, DAG edits).
- `_execute_ready_tasks` uses `asyncio.TaskGroup` for parallel execution of independent tasks.
- Critical sections are kept minimal to avoid contention.

### Testing

```bash
pytest                    # Run all tests
pytest test/test_agent.py  # Agent-specific tests
```

Tests mock LLM calls and test: task execution, tool behavior, YAML workflow import, URL fetching, checkpointing.

### Error handling

- Context overflow: detected via `_is_context_limit_error()`, triggers a retry with a compressed prompt that includes scratch notes as checkpoint.
- Token budget: checked at top of each loop iteration.
- No-progress guardrail: 3 consecutive cycles without executed tasks → abort with deadlock report.
- Max attempts per task: critic-driven; exceeded → `FAILED` status.

### Model provider resolution

- Bare string `"gpt-4.1-mini"` → OpenAI provider (default).
- Prefixed `"openai:..."` or `"anthropic:..."` → explicit provider.
- `Model` instance → used as-is (fully custom).
