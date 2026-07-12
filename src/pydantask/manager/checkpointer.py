# from asyncio import tasks
import json
import uuid
import asyncio

from pydantic import BaseModel, Field
from typing import Literal, Any, Dict
from datetime import datetime
from pathlib import Path

CheckpointEventType = Literal[
    "task_added",
    "task_patched",
    "task_status_updated",
    "task_result",
    "task_metadata_appended",
    "scratch_note_appended",
    "supervisor_decision",
    "critic_feedback",
    "final_task_set",
]


class CheckpointEvent(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now())
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: CheckpointEventType
    payload: Dict[str, Any] = Field(default_factory=dict)


class CheckpointRecorder:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path = directory / "events.jsonl"
        self.summary_path = directory / "summaries.jsonl"
        self._lock = asyncio.Lock()

    async def record(self, event_type: CheckpointEventType, payload: Dict[str, Any]) -> None:
        event = CheckpointEvent(type=event_type, payload=payload)
        await self._append_json_line(self.log_path, event.model_dump_json())

    async def record_summary(self, summary: Dict[str, Any]) -> None:
       await self._append_json_line(self.summary_path, json.dumps(summary))

    async def load_events(self) -> list[CheckpointEvent]:
        if not self.log_path.exists():
            return []

        def _read_lines() -> list[str]:
            with self.log_path.open("r", encoding="utf-8") as fh:
                return [line.strip() for line in fh if line.strip()]

        lines = await asyncio.to_thread(_read_lines)
        return [CheckpointEvent.model_validate_json(line) for line in lines]

    async def _append_json_line(self, path: Path, json_line: str) -> None:
        async with self._lock:
            def _write_line() -> None:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json_line + "\n")

            await asyncio.to_thread(_write_line)