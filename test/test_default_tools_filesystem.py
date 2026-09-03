from __future__ import annotations

from types import SimpleNamespace

import pytest

from pydantask.models import RuntimeState, TaskItem, TaskRunDeps, TaskStatus
from pydantask.tools import default_tools


@pytest.mark.asyncio
async def test_write_and_read_file_system_tools_work_with_runtime_state_deps(
    tmp_path, monkeypatch
):
    base_dir = tmp_path / "tmp_files"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(default_tools, "DEFAULT_DIR", base_dir)

    runtime = RuntimeState(objective="obj", capability_registry={}, next_task_id=1)
    ctx = SimpleNamespace(deps=runtime)

    msg = await default_tools.write_to_file_system(
        ctx, file_name="a.txt", content="hello", overwrite=True
    )
    assert "Content written" in msg
    assert runtime.document_store["a.txt"] == "a.txt"

    content = await default_tools.read_from_file_system(ctx, file_name="a.txt")
    assert "hello" in content


@pytest.mark.asyncio
async def test_write_and_read_file_system_tools_work_with_task_run_deps(
    tmp_path, monkeypatch
):
    base_dir = tmp_path / "tmp_files"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(default_tools, "DEFAULT_DIR", base_dir)

    runtime = RuntimeState(objective="obj", capability_registry={}, next_task_id=1)
    task = TaskItem(
        task_id=1,
        overall_objective="obj",
        sub_task_objective="do",
        capability="worker_agent",
        status=TaskStatus.READY,
    )
    deps = TaskRunDeps(runtime_state=runtime, task=task)
    ctx = SimpleNamespace(deps=deps)

    await default_tools.write_to_file_system(
        ctx, file_name="b.txt", content="world", overwrite=True
    )
    assert runtime.document_store["b.txt"] == "b.txt"

    content = await default_tools.read_from_file_system(ctx, file_name="b.txt")
    assert "world" in content


@pytest.mark.asyncio
async def test_save_and_read_task_context_use_shared_runtime_document_store(
    tmp_path, monkeypatch
):
    base_dir = tmp_path / "tmp_files"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(default_tools, "DEFAULT_DIR", base_dir)

    runtime = RuntimeState(objective="obj", capability_registry={}, next_task_id=1)
    task = TaskItem(
        task_id=7,
        overall_objective="obj",
        sub_task_objective="memo",
        capability="worker_agent",
        status=TaskStatus.READY,
    )
    deps = TaskRunDeps(runtime_state=runtime, task=task)
    ctx = SimpleNamespace(deps=deps)

    await default_tools.save_task_context(
        ctx, task_id=7, content="notes", kind="notes", overwrite=True
    )

    key = "task-7-notes.md"
    assert key in runtime.document_store

    read_back = await default_tools.read_task_context(ctx, task_id=7, kind="notes")
    assert "notes" in read_back
