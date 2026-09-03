from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

import pydantask.agents.agent as agent_mod
from pydantask.tools import default_tools
from pydantask.capabilities.runner_v2 import as_runner
from pydantask.models import (
    RuntimeState,
    TaskItem,
    TaskQAResult,
    TaskResult,
    TaskStatus,
    SupervisorDecision,
    TaskRunDeps,
)


class DummyAsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyRecorder:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.summaries: list[dict] = []
        self._events_to_load: list[agent_mod.CheckpointEvent] = []

    async def record(self, event_type, payload):
        self.events.append((event_type, payload))

    async def record_summary(self, summary):
        self.summaries.append(summary)

    async def load_events(self):
        return list(self._events_to_load)


@pytest.fixture(autouse=True)
def env_vars(monkeypatch: pytest.MonkeyPatch):
    # Keep init() happy if a test *does* instantiate DeepAgent.
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")


@pytest.fixture
def runtime_state() -> RuntimeState:
    return RuntimeState(objective="obj", capability_registry={}, next_task_id=1)


def make_minimal_deep_agent(prompt: str = "obj") -> agent_mod.DeepAgent:
    """Create a DeepAgent without running its heavy __init__.

    Since `__init__` is skipped, this function must define any attributes that
    methods under test expect to exist.
    """
    da = agent_mod.DeepAgent.__new__(agent_mod.DeepAgent)

    # Core run() expectations
    da.objective = prompt
    da._max_steps = 3
    da.token_budget = None
    da.verbose = False

    # Checkpoint/resume flags
    da.checkpoint = False
    da.resume = False
    da._checkpoint_recorder = None
    da.checkpoint_path = None

    # Planning / registry
    da.seed_plan = None
    da.planning_mode = "llm"
    da._capability_registry = {}

    # Concurrency + agents (mocked)
    da._plan_lock = DummyAsyncLock()
    da._last_scheduler_report = ""
    da._supervisor_agent = MagicMock()
    da._critic_agent = MagicMock()

    # Misc
    da._retry_model = MagicMock()

    return da


def test_autodetect_tracing_backend_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LOGFIRE_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    assert agent_mod.autodetect_tracing_backend() == agent_mod.TracingBackend.NONE

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert agent_mod.autodetect_tracing_backend() == agent_mod.TracingBackend.LANGSMITH

    monkeypatch.setenv("LOGFIRE_API_KEY", "x")
    assert agent_mod.autodetect_tracing_backend() == agent_mod.TracingBackend.LOGFIRE

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert agent_mod.autodetect_tracing_backend() == agent_mod.TracingBackend.LANGFUSE


def test_deep_agent_init_sets_registry_keys(monkeypatch: pytest.MonkeyPatch):
    # pydantic-ai inspects tools as callables and expects `__name__`.
    def _fake_tavily_tool(*args, **kwargs):
        return {"ok": True}

    with (
        patch.object(
            agent_mod.DeepAgent, "_create_retrying_client", return_value=AsyncClient()
        ),
        patch.object(agent_mod, "OpenAIProvider", autospec=True),
        patch.object(agent_mod, "OpenAIChatModel", autospec=True),
        patch.object(agent_mod, "tavily_search_tool", return_value=_fake_tavily_tool),
        # Avoid pulling in pydantic-ai's tool schema machinery for this unit test.
        patch.object(agent_mod, "Agent", autospec=True) as agent_cls,
    ):
        deep_agent = agent_mod.DeepAgent(
            "Test Goal", trace=False, default_capabilities_enabled=True
        )

    assert deep_agent.objective == "Test Goal"
    assert "research_agent" in deep_agent._capability_registry
    assert "producer_agent" in deep_agent._capability_registry

    # Soft-disabled filesystem tools should not be registered by default.
    all_tools: list[object] = []
    for c in agent_cls.call_args_list:
        tools = c.kwargs.get("tools") or []
        all_tools.extend(tools)

    all_tool_names = {getattr(t, "__name__", str(t)) for t in all_tools}
    assert "read_from_file_system" not in all_tool_names
    assert "write_to_file_system" not in all_tool_names
    assert "save_task_context" not in all_tool_names
    assert "read_task_context" not in all_tool_names


@pytest.mark.asyncio
async def test_add_cancel_patch_task(runtime_state: RuntimeState):
    da = make_minimal_deep_agent()
    ctx = SimpleNamespace(deps=runtime_state)

    task_id = await da.add_task(
        ctx,
        sub_task_objective="do the thing",
        capability="worker_agent",
        dependencies=[123],
        metadata={"k": "v"},
    )

    assert task_id == 1
    assert runtime_state.next_task_id == 2

    task = runtime_state.plan[task_id]
    assert task.status == TaskStatus.READY
    assert task.sub_task_dependencies == [123]
    assert task.metadata == {"k": "v"}

    msg = await da.patch_task(
        ctx, task_id=task_id, sub_task_objective="new", dependencies=[1, 2]
    )
    assert "updated successfully" in msg
    assert runtime_state.plan[task_id].sub_task_objective == "new"
    assert runtime_state.plan[task_id].sub_task_dependencies == [1, 2]

    msg = await da.cancel_task(ctx, task_id=task_id, reason="no longer needed")
    assert "cancelled" in msg
    assert runtime_state.plan[task_id].status == TaskStatus.CANCELLED


def test_dependencies_satisfied_only_completed(runtime_state: RuntimeState):
    da = make_minimal_deep_agent()

    t1 = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="dep",
        capability="worker_agent",
        status=TaskStatus.COMPLETED,
    )
    t2 = TaskItem(
        task_id=2,
        overall_objective=runtime_state.objective,
        sub_task_objective="blocked",
        capability="worker_agent",
        status=TaskStatus.READY,
        sub_task_dependencies=[1, 999],
    )

    runtime_state.plan = {1: t1, 2: t2}

    assert da._dependencies_satisfied(t1, runtime_state) is True
    assert da._dependencies_satisfied(t2, runtime_state) is False  # dep 999 missing

    runtime_state.plan[999] = TaskItem(
        task_id=999,
        overall_objective=runtime_state.objective,
        sub_task_objective="present but not complete",
        capability="worker_agent",
        status=TaskStatus.RUNNING,
    )
    assert da._dependencies_satisfied(t2, runtime_state) is False

    runtime_state.plan[999].status = TaskStatus.COMPLETED
    assert da._dependencies_satisfied(t2, runtime_state) is True


@pytest.mark.asyncio
async def test_handle_critic_result_transitions():
    da = make_minimal_deep_agent()

    task = TaskItem(
        task_id=1,
        overall_objective="obj",
        sub_task_objective="do it",
        capability="worker_agent",
        status=TaskStatus.NEEDS_REVIEW,
    )

    await da.handle_critic_result(
        task, TaskQAResult(task_id=1, passed=True, reasoning="ok")
    )
    assert task.status == TaskStatus.COMPLETED

    task2 = TaskItem(
        task_id=2,
        overall_objective="obj",
        sub_task_objective="do it",
        capability="worker_agent",
        status=TaskStatus.NEEDS_REVIEW,
        attempt_count=0,
        max_attempts=2,
    )
    await da.handle_critic_result(
        task2, TaskQAResult(task_id=2, passed=False, reasoning="not good")
    )
    assert task2.status == TaskStatus.RERUN
    assert task2.attempt_count == 1
    assert "Previous attempt failed review" in task2.sub_task_objective

    task2.attempt_count = 2
    await da.handle_critic_result(
        task2, TaskQAResult(task_id=2, passed=False, reasoning="still bad")
    )
    assert task2.status == TaskStatus.FAILED
    assert "Max retries reached" in (task2.error_msg or "")


@pytest.mark.asyncio
async def test_execute_sets_result_and_needs_review(runtime_state: RuntimeState):
    da = make_minimal_deep_agent(prompt="overall")

    sub_agent = MagicMock(name="sub_agent")
    sub_agent.run = AsyncMock(name="run")

    tr = TaskResult(task_id=1, summary="done")
    sub_agent.run.return_value = SimpleNamespace(output=tr)

    step = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="do",
        capability="worker_agent",
        status=TaskStatus.READY,
    )

    out_step = await da.execute(sub_agent, step, runtime_state)

    assert out_step.status == TaskStatus.NEEDS_REVIEW
    assert out_step.result == tr
    sub_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_ready_tasks_filters_deps_and_injects_feedback(
    runtime_state: RuntimeState,
):
    da = make_minimal_deep_agent(prompt="overall")

    # plan: task 1 completed, task 2 ready (depends on 1), task 3 blocked (depends on missing)
    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="dep",
        capability="worker_agent",
        status=TaskStatus.COMPLETED,
    )
    runtime_state.plan[2] = TaskItem(
        task_id=2,
        overall_objective=runtime_state.objective,
        sub_task_objective="ready",
        capability="worker_agent",
        status=TaskStatus.READY,
        sub_task_dependencies=[1],
    )
    runtime_state.plan[3] = TaskItem(
        task_id=3,
        overall_objective=runtime_state.objective,
        sub_task_objective="blocked",
        capability="worker_agent",
        status=TaskStatus.READY,
        sub_task_dependencies=[999],
    )

    worker_impl = MagicMock(name="worker_impl")
    da._capability_registry = {
        "worker_agent": agent_mod.CapabilityDescription(
            name="worker_agent",
            description="",
            tool_func=worker_impl,
        )
    }

    async def _execute_side_effect(sub_agent, step: TaskItem, ctx: RuntimeState):
        # mimic DeepAgent.execute returning the (mutated) step
        step.result = TaskResult(task_id=step.task_id, summary=f"ran {step.task_id}")
        step.status = TaskStatus.NEEDS_REVIEW
        return step

    da.execute = AsyncMock(side_effect=_execute_side_effect)

    decision = SupervisorDecision(
        reasoning="",
        tasks_to_execute=[2, 3],
        feedback_to_subagents={2: "focus on X"},
        all_tasks_completed=False,
    )

    results = await da._execute_ready_tasks(decision, runtime_state)

    assert [t.task_id for t in results] == [2]
    assert runtime_state.plan[2].parameters["supervisor_feedback"] == "focus on X"
    da.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_task_status_and_view_qa_report(runtime_state: RuntimeState):
    da = make_minimal_deep_agent(prompt="overall")
    ctx = SimpleNamespace(deps=runtime_state)

    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="x",
        capability="worker_agent",
        status=TaskStatus.READY,
    )

    msg = await da.update_task_status(ctx, task_id=1, status=TaskStatus.COMPLETED)
    assert "now" in msg
    assert runtime_state.plan[1].status == TaskStatus.COMPLETED

    runtime_state.plan[1].task_feedback = TaskQAResult(
        task_id=1, passed=True, reasoning="ok"
    )
    report = await da.view_qa_report(ctx, task_id=1)
    assert '"passed": true' in report


@pytest.mark.asyncio
async def test_add_task_emits_checkpoint_event_when_enabled(
    runtime_state: RuntimeState,
):
    da = make_minimal_deep_agent()
    recorder = DummyRecorder()
    da._checkpoint_recorder = recorder
    runtime_state.checkpoint_recorder = recorder
    ctx = SimpleNamespace(deps=runtime_state)

    task_id = await da.add_task(
        ctx,
        sub_task_objective="checkpoint",
        capability="worker_agent",
    )

    assert task_id == 1
    assert recorder.events and recorder.events[0][0] == "task_added"
    payload = recorder.events[0][1]
    assert payload["task"]["task_id"] == 1
    assert payload["next_task_id"] == runtime_state.next_task_id


@pytest.mark.asyncio
async def test_replay_checkpoint_rebuilds_state(runtime_state: RuntimeState):
    da = make_minimal_deep_agent()
    recorder = DummyRecorder()

    task_payload = TaskItem(
        task_id=3,
        overall_objective=runtime_state.objective,
        sub_task_objective="replayed",
        capability="worker_agent",
        status=TaskStatus.READY,
    ).model_dump()

    recorder._events_to_load = [
        agent_mod.CheckpointEvent(
            type="task_added",
            payload={"task": task_payload, "next_task_id": 4},
        ),
        agent_mod.CheckpointEvent(
            type="final_task_set",
            payload={"task_id": 3, "reason": "unit test"},
        ),
        agent_mod.CheckpointEvent(
            type="task_status_updated",
            payload={
                "task_id": 3,
                "status": TaskStatus.RUNNING.value,
                "reason": "claimed",
            },
        ),
        agent_mod.CheckpointEvent(
            type="critic_feedback",
            payload={
                "task_id": 3,
                "feedback": TaskQAResult(
                    task_id=3, passed=False, reasoning="fail"
                ).model_dump(),
                "attempt_count": 2,
            },
        ),
    ]

    da._checkpoint_recorder = recorder
    await da._replay_checkpoint(runtime_state)

    assert 3 in runtime_state.plan
    task = runtime_state.plan[3]
    assert task.is_final is True
    assert task.status == TaskStatus.RUNNING
    assert task.attempt_count == 2
    assert task.task_feedback and task.task_feedback.reasoning == "fail"
    assert task.metadata["status_history"][0]["reason"] == "claimed"
    assert runtime_state.next_task_id >= 4


@pytest.mark.asyncio
async def test_checkpoint_state_records_summary(runtime_state: RuntimeState):
    da = make_minimal_deep_agent()
    recorder = DummyRecorder()
    da._checkpoint_recorder = recorder

    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="x",
        capability="worker_agent",
        status=TaskStatus.READY,
    )

    await da._checkpoint_state(runtime_state)
    assert len(recorder.summaries) == 1
    summary = recorder.summaries[0]
    assert summary["total_tasks"] == 1
    assert summary["status_counts"][TaskStatus.READY.value] == 1


@pytest.mark.asyncio
async def test_append_scratch_note_records_checkpoint_event(
    runtime_state: RuntimeState,
):
    recorder = DummyRecorder()
    runtime_state.checkpoint_recorder = recorder

    task = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="memo",
        capability="worker_agent",
        status=TaskStatus.RUNNING,
        metadata={},
    )
    deps = TaskRunDeps(runtime_state=runtime_state, task=task)
    ctx = SimpleNamespace(deps=deps)

    await default_tools.append_scratch_note(ctx, task_id=1, note="remember this")

    assert "scratch_notes" in task.metadata
    assert recorder.events and recorder.events[-1][0] == "scratch_note_appended"
    assert recorder.events[-1][1]["task_id"] == 1


@pytest.mark.asyncio
async def test_get_task_result_tool_supports_runtime_state_deps(
    runtime_state: RuntimeState,
):
    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="x",
        capability="worker_agent",
        status=TaskStatus.COMPLETED,
        result=TaskResult(task_id=1, summary="s", detailed_output="d"),
    )

    ctx = SimpleNamespace(deps=runtime_state)
    out = await default_tools.get_task_result(ctx, task_id=1, max_chars=5_000)
    assert '"summary":' in out
    assert '"s"' in out


@pytest.mark.asyncio
async def test_list_completed_tasks_tool_supports_runtime_state_deps(
    runtime_state: RuntimeState,
):
    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="x",
        capability="worker_agent",
        status=TaskStatus.COMPLETED,
        result=TaskResult(task_id=1, summary="done"),
    )
    runtime_state.plan[2] = TaskItem(
        task_id=2,
        overall_objective=runtime_state.objective,
        sub_task_objective="y",
        capability="worker_agent",
        status=TaskStatus.READY,
    )

    ctx = SimpleNamespace(deps=runtime_state)
    out = await default_tools.list_completed_tasks(ctx)
    assert "task_id: 1" in out
    assert "task_id: 2" not in out


@pytest.mark.asyncio
async def test_run_stops_when_supervisor_says_done(runtime_state: RuntimeState):
    da = make_minimal_deep_agent(prompt="overall")

    # Pre-populate the runtime with a completed final task so the deterministic
    # completion guardrail accepts the supervisor's completion signal.
    final_result = TaskResult(task_id=1, summary="final", detailed_output="ok")
    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="final",
        capability="worker_agent",
        status=TaskStatus.COMPLETED,
        result=final_result,
        is_final=True,
    )

    da._initialize_runtime_state = MagicMock(return_value=runtime_state)
    da._format_supervisor_input_prompt = MagicMock(return_value="prompt")

    supervisor = MagicMock(name="supervisor")
    supervisor.run = AsyncMock(
        return_value=SimpleNamespace(
            output=SupervisorDecision(
                reasoning="done",
                tasks_to_execute=[],
                feedback_to_subagents=None,
                all_tasks_completed=True,
            )
        )
    )
    da._supervisor_agent = supervisor

    result = await da.run()

    assert result.objective == "overall"
    assert result.final_result is not None
    assert result.final_result.summary == "final"
    assert result.plan == runtime_state.plan
    supervisor.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_mark_final_task_sets_flag_and_emits_event(runtime_state: RuntimeState):
    da = make_minimal_deep_agent(prompt="overall")
    recorder = DummyRecorder()
    da._checkpoint_recorder = recorder

    # Two tasks, mark the second as final.
    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="t1",
        capability="worker_agent",
        status=TaskStatus.READY,
        is_final=True,
    )
    runtime_state.plan[2] = TaskItem(
        task_id=2,
        overall_objective=runtime_state.objective,
        sub_task_objective="t2",
        capability="worker_agent",
        status=TaskStatus.READY,
        is_final=False,
    )

    ctx = SimpleNamespace(deps=runtime_state)
    msg = await da.mark_final_task(ctx, task_id=2, reason="unit")

    assert "marked as final" in msg
    assert runtime_state.plan[1].is_final is False
    assert runtime_state.plan[2].is_final is True

    assert recorder.events
    evt_type, payload = recorder.events[-1]
    assert evt_type == "final_task_set"
    assert payload["task_id"] == 2


@pytest.mark.asyncio
async def test_run_overrides_completion_when_no_final_task(runtime_state: RuntimeState):
    da = make_minimal_deep_agent(prompt="overall")
    da._initialize_runtime_state = MagicMock(return_value=runtime_state)
    da._format_supervisor_input_prompt = MagicMock(return_value="prompt")

    supervisor = MagicMock(name="supervisor")
    supervisor.run = AsyncMock(
        return_value=SimpleNamespace(
            output=SupervisorDecision(
                reasoning="done",
                tasks_to_execute=[],
                feedback_to_subagents=None,
                all_tasks_completed=True,
            )
        )
    )
    da._supervisor_agent = supervisor

    result = await da.run()

    # The completion guardrail prevents early stop; we run until max_steps.
    assert supervisor.run.await_count == da._max_steps
    assert result.final_result is None


@pytest.mark.asyncio
async def test_scheduler_marks_callable_task_errored_when_missing_parameters(
    runtime_state: RuntimeState,
):
    da = make_minimal_deep_agent(prompt="overall")

    async def write_something_to_file(content: str, filename: str) -> str:
        return f"wrote {filename}"

    da._capability_registry = {
        "write_to_file": agent_mod.CapabilityDescription(
            name="write_to_file",
            description="",
            tool_func=as_runner(write_something_to_file),
        )
    }

    # Missing the required 'filename'
    runtime_state.plan[1] = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="write a file",
        capability="write_to_file",
        status=TaskStatus.READY,
        parameters={"content": "hello"},
    )

    report = await da._scheduler_pass(runtime_state)

    assert "missing required parameters" in report.lower()
    assert runtime_state.plan[1].status == TaskStatus.ERRORED
    assert runtime_state.plan[1].metadata.get("missing_parameters") == ["filename"]
    assert (
        "missing required parameters" in (runtime_state.plan[1].error_msg or "").lower()
    )


@pytest.mark.asyncio
async def test_coerce_output_ingests_existing_file_as_artifact(
    tmp_path, runtime_state: RuntimeState
):
    da = make_minimal_deep_agent(prompt="overall")

    # Create a fake checkpoint recorder so artifacts go under tmp_path.
    cp_dir = tmp_path / "cp"
    cp_dir.mkdir(parents=True, exist_ok=True)

    class _Recorder:
        def __init__(self, directory):
            self.directory = directory

        async def record(self, event_type, payload):
            return None

    runtime_state.checkpoint_recorder = _Recorder(cp_dir)

    # Create an output file the callable might have produced.
    out_file = cp_dir / "haiku.md"
    out_file.write_text(
        "stars drift\nengines hum softly\nhome is far away\n", encoding="utf-8"
    )

    step = TaskItem(
        task_id=1,
        overall_objective=runtime_state.objective,
        sub_task_objective="write a haiku file",
        capability="write_to_file",
        status=TaskStatus.RUNNING,
    )
    runtime_state.plan[1] = step

    tr = await da._coerce_output_to_task_result(
        step, str(out_file), runtime_state=runtime_state
    )

    assert isinstance(tr, TaskResult)
    assert tr.artifacts and len(tr.artifacts) == 1
    assert tr.artifacts[0].artifact_id.startswith("sha256:")
    assert tr.artifacts[0].uri.startswith("artifacts/")
    assert "file outputs ingested as artifacts" in tr.detailed_output.lower()
    assert "stars drift" in tr.detailed_output
