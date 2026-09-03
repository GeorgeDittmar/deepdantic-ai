from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
import mimetypes

from loguru import logger
from pydantic_ai import RunContext

from pydantask.models import ArtifactRef, RuntimeState, TaskRunDeps

ARTIFACT_DIRNAME = "artifacts"
DEFAULT_MAX_PREVIEW_CHARS = 500


def _guess_mime_type_for_path(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


async def store_file_as_artifact(
    runtime: RuntimeState,
    *,
    file_path: str | Path,
    task_id: int | None,
    name: str | None = None,
    mime_type: str | None = None,
    max_preview_chars: int = DEFAULT_MAX_PREVIEW_CHARS,
) -> ArtifactRef:
    """Persist a local file into the run's artifact store and return an ArtifactRef.

    This is intended for *host-side* orchestration code (e.g. DeepAgent) that
    needs to convert arbitrary callable outputs (like "wrote tmp/foo.txt") into
    durable, resumable artifacts.

    Notes:
      - This is NOT a tool; agents should use `put_artifact` instead.
      - The file is read and stored content-addressed (sha256) under the
        checkpoint artifact root when available.
      - A record is attached to the owning TaskItem metadata under `artifacts`
        (for audit/replay parity with the `put_artifact` tool).
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {str(p)}")

    mt = (mime_type or "").strip() or _guess_mime_type_for_path(p)

    raw = await asyncio.to_thread(p.read_bytes)
    sha256 = hashlib.sha256(raw).hexdigest()
    artifact_id = f"sha256:{sha256}"

    root = _artifact_root(runtime)
    ext = _ext_for_mime(mt)

    filename = sha256 + ext
    rel_uri = f"{ARTIFACT_DIRNAME}/{filename}"
    path = root / filename
    meta_path = root / (sha256 + ".meta.json")

    safe_label = _safe_name(name) or _safe_name(p.name)

    def _write() -> None:
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp." + uuid.uuid4().hex)
            tmp.write_bytes(raw)
            os.replace(tmp, path)

        meta = {
            "artifact_id": artifact_id,
            "uri": rel_uri,
            "name": safe_label,
            "mime_type": mt,
            "size_bytes": len(raw),
            "sha256": sha256,
            "created_at": datetime.now().isoformat(),
            "task_id": task_id,
            "source_path": str(p),
        }
        tmpm = meta_path.with_suffix(meta_path.suffix + ".tmp." + uuid.uuid4().hex)
        tmpm.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmpm, meta_path)

    await asyncio.to_thread(_write)

    preview: str | None = None
    try:
        if mt.lower().startswith("text/") or mt.lower() in {
            "application/json",
            "application/yaml",
            "text/yaml",
            "text/csv",
            "application/csv",
        }:
            preview = raw.decode("utf-8", errors="replace")
            preview = _truncate_text(preview, max_chars=max_preview_chars)
    except Exception:
        preview = None

    ref_payload: dict[str, Any] = {
        "artifact_id": artifact_id,
        "uri": rel_uri,
        "name": safe_label,
        "mime_type": mt,
        "size_bytes": len(raw),
        "sha256": sha256,
        "preview": preview,
    }

    # Attach to task metadata for discoverability.
    if task_id is not None:
        task = runtime.plan.get(int(task_id)) if runtime.plan else None
        if task is not None:
            task.metadata.setdefault("artifacts", [])
            if isinstance(task.metadata.get("artifacts"), list):
                task.metadata["artifacts"].append(
                    {**ref_payload, "task_id": int(task_id)}
                )
            else:
                task.metadata["artifacts"] = [{**ref_payload, "task_id": int(task_id)}]

            recorder = getattr(runtime, "checkpoint_recorder", None)
            if recorder is not None:
                try:
                    await recorder.record(
                        "task_metadata_appended",
                        {
                            "task_id": int(task_id),
                            "key": "artifacts",
                            "value": {**ref_payload, "task_id": int(task_id)},
                        },
                    )
                except Exception:
                    pass

    return ArtifactRef.model_validate(ref_payload)


def _truncate_text(text: str, max_chars: int | None) -> str:
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


def _get_runtime_state(deps: RuntimeState | TaskRunDeps) -> RuntimeState:
    if isinstance(deps, RuntimeState):
        return deps
    return deps.runtime_state


def _guess_task_id(deps: RuntimeState | TaskRunDeps, task_id: int | None) -> int | None:
    """Return the active task_id if available.

    - For worker/research tasks, deps is TaskRunDeps and `deps.task.task_id` is authoritative.
    - For supervisor tools, deps is RuntimeState and task_id must be passed explicitly.
    """
    if isinstance(deps, TaskRunDeps):
        active = int(deps.task.task_id)
        if task_id is not None and int(task_id) != active:
            # Avoid silently attributing artifacts to the wrong task.
            raise ValueError(
                f"task_id mismatch: got task_id={task_id}, active_task_id={active}"
            )
        return active
    return int(task_id) if task_id is not None else None


def _safe_name(name: str | None) -> str | None:
    if not name:
        return None
    # Keep names short and filesystem-safe.
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9._\- ]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name[:120] if name else None


def _ext_for_mime(mime_type: str) -> str:
    mt = (mime_type or "").lower().strip()
    if mt in {"application/json"}:
        return ".json"
    if mt in {"text/markdown", "text/md"}:
        return ".md"
    if mt.startswith("text/"):
        return ".txt"
    if mt in {"application/yaml", "text/yaml"}:
        return ".yaml"
    if mt in {"text/csv", "application/csv"}:
        return ".csv"
    return ".bin"


def _artifact_root(runtime_state: RuntimeState) -> Path:
    """Resolve the artifact root directory.

    Preference order:
    1) Under the active checkpoint directory (if checkpointing enabled)
    2) A local fallback under `src/pydantask/tools/tmp_files/artifacts/`

    Agents do NOT control the directory.
    """
    recorder = getattr(runtime_state, "checkpoint_recorder", None)
    directory = getattr(recorder, "directory", None)
    if isinstance(directory, Path):
        root = directory / ARTIFACT_DIRNAME
        root.mkdir(parents=True, exist_ok=True)
        return root

    # Fallback: repo-local temp dir
    base_dir = Path(__file__).parent.resolve() / "tmp_files" / ARTIFACT_DIRNAME
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


async def put_artifact(
    ctx: RunContext[RuntimeState | TaskRunDeps],
    *,
    content: str,
    name: str | None = None,
    mime_type: str = "text/plain",
    content_is_base64: bool = False,
    task_id: int | None = None,
    max_preview_chars: int = DEFAULT_MAX_PREVIEW_CHARS,
) -> str:
    """Tool: Put Artifact (durable, segregated filestore).

    This is a safer alternative to exposing raw filesystem tools to agents.

    - Agents do NOT choose file paths.
    - Content is stored under the run's checkpoint directory when available.
    - The tool returns a small JSON "ArtifactRef" containing an ID + URI.

    Args:
        content: Artifact content as text (utf-8) or base64-encoded bytes.
        name: Optional human-friendly label.
        mime_type: MIME type (e.g. "application/json", "text/markdown").
        content_is_base64: If True, treat `content` as base64-encoded bytes.
        task_id: Optional explicit task_id for supervisor-context calls.
        max_preview_chars: How much preview text to return.

    Returns:
        JSON string describing the stored artifact.
    """
    runtime = _get_runtime_state(ctx.deps)

    try:
        resolved_task_id = _guess_task_id(ctx.deps, task_id)
    except Exception as e:
        return f"Error: {str(e)}"

    safe_label = _safe_name(name)

    # Normalize bytes
    if content_is_base64:
        try:
            raw = base64.b64decode(content.encode("utf-8"), validate=True)
        except Exception:
            return "Error: content_is_base64=True but content is not valid base64."
    else:
        raw = (content or "").encode("utf-8")

    sha256 = hashlib.sha256(raw).hexdigest()
    artifact_id = f"sha256:{sha256}"

    root = _artifact_root(runtime)
    ext = _ext_for_mime(mime_type)

    # Use content-addressed storage for dedupe.
    filename = sha256 + ext
    rel_uri = f"{ARTIFACT_DIRNAME}/{filename}"
    path = root / filename

    meta_path = root / (sha256 + ".meta.json")

    def _write() -> None:
        # Avoid rewriting if already present (dedupe).
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp." + uuid.uuid4().hex)
            tmp.write_bytes(raw)
            os.replace(tmp, path)

        # Write/update metadata (best-effort)
        meta = {
            "artifact_id": artifact_id,
            "uri": rel_uri,
            "name": safe_label,
            "mime_type": mime_type,
            "size_bytes": len(raw),
            "sha256": sha256,
            "created_at": datetime.now().isoformat(),
            "task_id": resolved_task_id,
        }
        tmpm = meta_path.with_suffix(meta_path.suffix + ".tmp." + uuid.uuid4().hex)
        tmpm.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmpm, meta_path)

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        logger.exception("Failed to write artifact")
        return f"Error: failed to persist artifact: {str(e)}"

    preview: str | None
    try:
        if mime_type.lower().startswith("text/") or mime_type.lower() in {
            "application/json",
            "application/yaml",
            "text/yaml",
            "text/csv",
            "application/csv",
        }:
            preview = raw.decode("utf-8", errors="replace")
            preview = _truncate_text(preview, max_chars=max_preview_chars)
        else:
            preview = None
    except Exception:
        preview = None

    ref: dict[str, Any] = {
        "artifact_id": artifact_id,
        "uri": rel_uri,
        "name": safe_label,
        "mime_type": mime_type,
        "size_bytes": len(raw),
        "sha256": sha256,
        "preview": preview,
        "task_id": resolved_task_id,
    }

    # Attach to task metadata when possible (and checkpoint it via existing event type).
    if resolved_task_id is not None:
        task = runtime.plan.get(resolved_task_id) if runtime.plan else None
        if task is not None:
            task.metadata.setdefault("artifacts", [])
            if isinstance(task.metadata.get("artifacts"), list):
                task.metadata["artifacts"].append(ref)
            else:
                task.metadata["artifacts"] = [ref]

            recorder = getattr(runtime, "checkpoint_recorder", None)
            if recorder is not None:
                # Use the existing metadata append event type for replay.
                try:
                    await recorder.record(
                        "task_metadata_appended",
                        {"task_id": resolved_task_id, "key": "artifacts", "value": ref},
                    )
                except Exception:
                    # Best-effort; do not fail the tool.
                    pass

    return json.dumps(ref, ensure_ascii=False)


async def get_artifact(
    ctx: RunContext[RuntimeState | TaskRunDeps],
    *,
    artifact_id: str | None = None,
    uri: str | None = None,
    max_chars: int | None = 10_000,
) -> str:
    """Tool: Get Artifact (bounded read).

    Provide either `artifact_id` (preferred) or `uri`.

    Returns text-like content only. Binary artifacts return a short message.
    """
    runtime = _get_runtime_state(ctx.deps)
    root = _artifact_root(runtime)

    if not artifact_id and not uri:
        return "Error: provide either artifact_id or uri."

    # Map to path
    path: Path | None = None
    if uri:
        # Expect checkpoint-relative like "artifacts/<name>"; strip any leading slashes.
        u = uri.lstrip("/\\")
        if not u.startswith(f"{ARTIFACT_DIRNAME}/"):
            return f"Error: uri must start with '{ARTIFACT_DIRNAME}/'."
        path = root.parent / u  # root is .../artifacts; root.parent is checkpoint dir

    if path is None and artifact_id:
        # artifact_id is "sha256:<hex>"
        if not artifact_id.startswith("sha256:"):
            return "Error: only sha256:<hex> artifact_id is supported by this store."
        hexhash = artifact_id.split(":", 1)[1]
        # We don't know extension; try common ones and then any match.
        for ext in (".json", ".md", ".txt", ".yaml", ".csv", ".bin"):
            candidate = root / (hexhash + ext)
            if candidate.exists():
                path = candidate
                break
        if path is None:
            # Try glob for any extension.
            matches = list(root.glob(hexhash + ".*"))
            if matches:
                path = matches[0]

    if path is None or not path.exists():
        return "Error: artifact not found."

    # If it's obviously binary, do not dump it into context.
    if path.suffix.lower() in {".bin"}:
        size = path.stat().st_size
        return f"Artifact is binary ({path.name}, {size} bytes). Use it via its uri/id rather than reading into context."

    try:
        text = await asyncio.to_thread(path.read_text, "utf-8", "replace")
    except Exception:
        try:
            data = await asyncio.to_thread(path.read_bytes)
            text = data.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error: failed to read artifact: {str(e)}"

    return _truncate_text(text, max_chars=max_chars)


async def list_artifacts(
    ctx: RunContext[RuntimeState | TaskRunDeps],
    *,
    task_id: int | None = None,
    max_items: int = 50,
) -> str:
    """Tool: List Artifacts.

    If task_id is provided, returns artifacts recorded on that task's metadata.
    Otherwise, lists artifacts on the *active* task (if any).

    Note: this does not enumerate the artifact directory on disk; it lists the
    artifacts the runtime has associated with tasks.
    """
    runtime = _get_runtime_state(ctx.deps)

    # Resolve task
    resolved_task_id: int | None
    try:
        resolved_task_id = _guess_task_id(ctx.deps, task_id)
    except Exception as e:
        return f"Error: {str(e)}"

    if resolved_task_id is None:
        return "Error: task_id is required when calling list_artifacts from supervisor context."

    task = runtime.plan.get(resolved_task_id) if runtime.plan else None
    if task is None:
        return f"Error: no task with id {resolved_task_id}."

    artifacts = task.metadata.get("artifacts")
    if not artifacts:
        return f"No artifacts recorded for task {resolved_task_id}."

    if not isinstance(artifacts, list):
        return f"Error: task {resolved_task_id} metadata['artifacts'] is not a list."

    items = artifacts[:max_items]
    lines: list[str] = [
        f"Artifacts for task {resolved_task_id} (showing {len(items)}/{len(artifacts)}):"
    ]
    for i, a in enumerate(items, start=1):
        if not isinstance(a, dict):
            continue
        lines.append(
            "\n".join(
                [
                    f"- {i}. {a.get('name') or '<unnamed>'}",
                    f"  artifact_id: {a.get('artifact_id')}",
                    f"  uri: {a.get('uri')}",
                    f"  mime_type: {a.get('mime_type')}",
                    f"  size_bytes: {a.get('size_bytes')}",
                ]
            )
        )

    return "\n".join(lines)


async def attach_artifact_to_result(
    ctx: RunContext[TaskRunDeps],
    *,
    artifact_ref: str | dict[str, Any],
    task_id: int | None = None,
) -> str:
    """Tool: Attach Artifact To Result (deferred).

    During an agent run, `TaskResult` is only returned at the end, so tools
    cannot mutate the final TaskResult object directly.

    This tool records a validated ArtifactRef into task metadata under
    `result_artifacts`. The orchestrator merges these into
    `TaskResult.artifacts` automatically after the agent run completes.

    Typical usage:
      1) ref_json = put_artifact(...)
      2) attach_artifact_to_result(artifact_ref=ref_json)

    Args:
        artifact_ref: Either a dict or a JSON string returned by `put_artifact`.
        task_id: Optional explicit task id; must match the active task id.

    Returns:
        A short confirmation message.
    """
    if task_id is not None and int(task_id) != int(ctx.deps.task.task_id):
        return (
            "Error: task_id does not match the active task. "
            f"Got task_id={task_id}, active_task_id={ctx.deps.task.task_id}."
        )

    if isinstance(artifact_ref, str):
        try:
            parsed = json.loads(artifact_ref)
        except Exception:
            return "Error: artifact_ref is a string but is not valid JSON."
    elif isinstance(artifact_ref, dict):
        parsed = artifact_ref
    else:
        return "Error: artifact_ref must be a JSON string or a dict."

    try:
        validated = ArtifactRef.model_validate(parsed)
    except Exception as e:
        return f"Error: artifact_ref is not a valid ArtifactRef: {str(e)}"

    payload = validated.model_dump(mode="json")

    key = "result_artifacts"
    ctx.deps.task.metadata.setdefault(key, [])
    existing = ctx.deps.task.metadata.get(key)
    if isinstance(existing, list):
        existing.append(payload)
    else:
        ctx.deps.task.metadata[key] = [payload]

    recorder = getattr(ctx.deps.runtime_state, "checkpoint_recorder", None)
    if recorder is not None:
        try:
            await recorder.record(
                "task_metadata_appended",
                {"task_id": ctx.deps.task.task_id, "key": key, "value": payload},
            )
        except Exception:
            pass

    return (
        "Attached artifact ref for inclusion in TaskResult.artifacts. "
        "(It will be merged automatically after this task finishes.)"
    )
