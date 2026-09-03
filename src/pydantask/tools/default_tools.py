from __future__ import annotations

from pathlib import Path
import ipaddress
import socket
from urllib.parse import urlparse
import asyncio
import httpx
from loguru import logger
from pydantic_ai import RunContext

from pydantask.models import RuntimeState, TaskRunDeps

BASE_DIR = Path(__file__).parent.resolve()  # Directory where this script is
DEFAULT_DIR = BASE_DIR / "tmp_files"  # TODO: make this configurable
DEFAULT_DIR.mkdir(parents=True, exist_ok=True)


def _truncate_text(text: str, max_chars: int | None) -> str:
    """Best-effort truncation helper to reduce tool output size.

    This is primarily used to avoid blowing up model context windows (common with
    smaller local models).

    If truncation occurs, we keep both the head and tail of the text.
    """
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
    """Return the RuntimeState regardless of whether deps is RuntimeState or TaskRunDeps."""
    if isinstance(deps, RuntimeState):
        return deps
    return deps.runtime_state


async def ask_user(ctx: RunContext[RuntimeState], question_for_user: str) -> str:
    """Prompt the user for input. This is a synchronous blocking call."""
    return await asyncio.to_thread(input, f"{question_for_user}: ")


async def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on progress and decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: str Your detailed reflection on research progress, findings, gaps, and next steps
    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f""


async def write_to_file_system(
    ctx: RunContext[RuntimeState | TaskRunDeps],
    file_name: str,
    content: str,
    overwrite: bool = False,
) -> str:
    """Write content to a file in the agent's workspace filesystem.

    This tool is designed to work with both:
    - `deps_type=RuntimeState` (supervisor-style tools), and
    - `deps_type=TaskRunDeps` (worker/research/producer tasks)

    Regardless of deps type, the *logical* file key is recorded in
    `RuntimeState.document_store` so other agents can discover and read it.

    IMPORTANT:
        - `file_name` must be a logical name (e.g. `agent_notes.md`), not a full path.
        - Content is appended by default; pass `overwrite=True` to replace.

    Args:
        file_name: Logical file name key.
        content: Text content to write.
        overwrite: If True, overwrite the file instead of appending.

    Returns:
        A confirmation message including how to read it back.
    """
    # Ensure the base directory exists
    # At top:

    path = DEFAULT_DIR / file_name
    mode = "w" if overwrite else "a"
    logger.info(f"Writing to file system at {path} with content: {content}")
    with open(path, mode, encoding="utf-8") as f:
        f.write(content + "\n")

    runtime = _get_runtime_state(ctx.deps)
    # Store by the logical file_name key so agents can read it back with that name
    runtime.document_store[file_name] = file_name

    return (
        f"Content written to {path}.\n"
        f"To read this file later, call read_from_file_system with file_name='{file_name}'. "
        f"Use exactly this file_name, not the full path."
    )


async def delete_from_file_system(path: str) -> str:
    """Delete a file or directory on the file system.

    Args: ""
        path: str = The path to the file or directory to delete.
    Returns:
        Confirmation message indicating deletion attempt.
    """
    # Attempt to delete the file, ignore error if it doesn't exist (Python 3.8+)
    try:
        DEFAULT_DIR.joinpath(path).unlink(missing_ok=True)
        return f"File '{path}' deletion attempted (if existed)."
    except PermissionError:
        return f"Permission denied to delete the file '{path}'."
    except Exception as e:
        return f"An error occurred: {e}"


async def read_file_contents(file_path:str):
    """Reads a file from the file system to inspect its contents. 
    
    When to use:
        - Need to read a file that is mentioned in a TaskResult"""
    
    with open(file_path, "r") as f:
        return f.read()

async def read_from_file_system(
    ctx: RunContext[RuntimeState | TaskRunDeps],
    file_name: str,
) -> str:
    """Read a previously written file by logical name.

    Looks up `file_name` in `RuntimeState.document_store` first (if present), then
    falls back to reading `DEFAULT_DIR / file_name`.

    Works with both `deps_type=RuntimeState` and `deps_type=TaskRunDeps`.

    Args:
        file_name: Logical file name key (recommended) or raw filename.

    Returns:
        File contents as a string, or an informative error message.
    """
    try:
        runtime = _get_runtime_state(ctx.deps)
        # First try lookup in doc store, fallback to the file_name
        logical_file = runtime.document_store.get(file_name, file_name)
        path = DEFAULT_DIR / logical_file
        logger.info(
            f"Attempting to read file with logical name '{file_name}' from document store. Found path: {path}"
        )

        full_path = Path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    except FileNotFoundError as e:
        runtime = _get_runtime_state(ctx.deps)
        existing = ", ".join(runtime.document_store.keys()) or "<none>"
        return (
            f"File '{file_name}' not found at path '{path}'.\n"
            f"Known document keys: {existing}\n"
            f"If you expected this to exist, you must first write it using write_to_file_system."
        )


# Tools used by the supervisor or researcher mostly
async def get_current_datetime() -> str:
    """
    Get the current date and time in ISO 8601 format.

    When to use:
        - Needing to look up current date and time for context to complete a task
        - Needing to timestamp a step or task
        - Comparing results between different times

    Args: None
    Returns:
        Current date and time as an ISO 8601 formatted string.
    """
    from datetime import datetime

    return str(datetime.now().isoformat())


async def list_documents(ctx: RunContext[RuntimeState | TaskRunDeps]) -> str:
    """List all documents that have been written in this run.

    Documents are tracked by logical key in `RuntimeState.document_store`.

    Returns:
        A human-readable list of logical names and their filesystem paths.
    """
    runtime = _get_runtime_state(ctx.deps)

    if not runtime.document_store:
        return "No documents have been written yet."

    lines: list[str] = []
    for name, logical in runtime.document_store.items():
        path = DEFAULT_DIR / logical  # Convert logical name to full path for display
        lines.append(f"- name: {name}\n  path: {path}")
    return "\n".join(lines)


async def list_completed_tasks(ctx: RunContext[RuntimeState | TaskRunDeps]) -> str:
    """List all tasks that have completed, with brief summaries.

    Works in both supervisor/critic context (deps_type=RuntimeState) and
    worker context (deps_type=TaskRunDeps).
    """
    runtime = _get_runtime_state(ctx.deps)
    if not runtime.plan:
        return "No tasks in plan."

    from pydantask.models import TaskStatus  # local import to avoid cycles

    lines: list[str] = []
    for task_id, task in sorted(runtime.plan.items(), key=lambda kv: kv[0]):
        if task.status != TaskStatus.COMPLETED:
            continue
        summary = task.result.summary if task.result is not None else "<no result>"
        lines.append(
            f"- task_id: {task_id}\n"
            f"  capability: {task.capability}\n"
            f"  objective: {task.sub_task_objective}\n"
            f"  summary: {summary}"
        )

    if not lines:
        return "No completed tasks yet."

    return "\n".join(lines)


async def save_task_context(
    ctx: RunContext[TaskRunDeps],
    task_id: int,
    content: str,
    kind: str = "notes",
    overwrite: bool = False,
) -> str:
    """Save contextual information for a specific task to a canonical file.

    FILENAME CONVENTION (IMPORTANT):
      - The file name will always be ``task-{task_id}-{kind}.md``.
      - Agents should NOT invent their own names when saving task context.

    When to use:
      - After completing a task, to save detailed notes or results for that task.
      - When you want to offload information from memory but keep it accessible by task_id.
      - To create a record of your thought process and findings for each task.

    Args:
        task_id: The ID of the task whose context you want to save.
        content: The string content to save, such as notes, findings, or detailed results.
        kind: A short label like "notes", "research", or "final"; it becomes part of the file name.
        overwrite: If True, overwrite the file instead of appending. Default is False (append).

    Returns:
    Confirmation message indicating where the content was saved and how to read it back.
    """
    file_name = f"task-{task_id}-{kind}.md"
    return await write_to_file_system(
        ctx, file_name=file_name, content=content, overwrite=overwrite
    )


async def read_task_context(
    ctx: RunContext[TaskRunDeps],
    task_id: int,
    kind: str = "notes",
) -> str:
    """Read contextual information for a specific task from its canonical file.

    This uses the same convention as ``save_task_context``:
      - file_name = ``task-{task_id}-{kind}.md``.

    Args:
        task_id: The task id whose context file you want to read.
        kind: The same kind string that was used when saving (e.g. "notes", "research", "final").

    Returns:
        File contents as a string, or an informative error message if it does not exist.
    """
    file_name = f"task-{task_id}-{kind}.md"
    return await read_from_file_system(ctx, file_name=file_name)


async def get_task_result(
    ctx: RunContext[RuntimeState | TaskRunDeps],
    task_id: int,
    max_chars: int | None = 20_000,
) -> str:
    """Return the full TaskResult for a given task_id as JSON.

    Works in both supervisor/critic context (deps_type=RuntimeState) and
    worker context (deps_type=TaskRunDeps).
    """

    runtime = _get_runtime_state(ctx.deps)

    if task_id not in runtime.plan:
        return (
            f"CRITICAL: task id: {task_id} does not exist in the plan. "
            "Attempt call again and double check the task id before trying again."
        )

    task = runtime.plan.get(task_id, None)

    if task is None:
        return f"CRITICAL: No task with id {task_id}."

    if task.result is None:
        return f"Task {task_id} has no result yet. Current status: {task.status}."

    result_json = task.result.model_dump_json(indent=2)
    return _truncate_text(result_json, max_chars=max_chars)


async def append_scratch_note(
    ctx: RunContext[TaskRunDeps],
    note: str,
    task_id: int | None = None,
) -> str:
    """
    Tool: Append Scratch Note
    Description: Append a short note to the tasks notepad / memory

    When to use:
        - You want to store intermediate notes or thoughts about your work and any.
    When not to use:
        - Writing final full reports / analysis or answers.
    """
    # Backwards-compatible: some callers pass task_id explicitly.
    # The authoritative task id is `ctx.deps.task.task_id`.
    if task_id is not None and int(task_id) != int(ctx.deps.task.task_id):
        return (
            "Error: task_id does not match the active task. "
            f"Got task_id={task_id}, active_task_id={ctx.deps.task.task_id}."
        )

    key = "scratch_notes"
    existing = ctx.deps.task.metadata.get(key, "")
    ctx.deps.task.metadata[key] = existing + f"\n\n{note}"

    recorder = getattr(ctx.deps.runtime_state, "checkpoint_recorder", None)
    if recorder is not None:
        await recorder.record(
            "scratch_note_appended",
            {
                "task_id": ctx.deps.task.task_id,
                "note": _truncate_text(note, max_chars=1_000),
            },
        )

    return f"Appended note to scratchpad {key}"


async def read_scratch_notes(
    ctx: RunContext[TaskRunDeps],
    max_chars: int | None = 8_000,
):
    """
    Tool: Read Scratch Notes
    Description: Reads any notes in the in-memory scratchpad for this task. Use to see if there are any thoughts you need to reason over.
    """

    key = f"scratch_notes"
    existing = ctx.deps.task.metadata.get(key, "")
    return _truncate_text(existing, max_chars=max_chars)


def _host_looks_local_or_private(host: str) -> tuple[bool, str]:
    """Best-effort SSRF guard.

    Blocks localhost and private/link-local/reserved IP space.

    Notes:
      - This is intentionally conservative.
      - For production-hardening you'd also want to consider DNS rebinding,
        allowlists, and/or running this behind an egress proxy.
    """
    h = (host or "").strip().lower()
    if not h:
        return True, "empty host"

    if h in {"localhost"}:
        return True, "localhost is not allowed"

    # If host is an IP literal, validate directly.
    try:
        ip = ipaddress.ip_address(h)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True, f"IP address {ip} is not allowed"
        return False, ""
    except ValueError:
        pass

    # Otherwise resolve DNS and validate the resulting IP(s).
    try:
        infos = socket.getaddrinfo(h, None)
    except Exception:
        # If we can't resolve, let the HTTP client handle the error.
        return False, ""

    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True, f"resolved IP address {ip} for host {h!r} is not allowed"

    return False, ""


async def fetch_url_content(
    url: str,
    max_chars: int | None = 20_000,
    *,
    timeout_s: float = 20.0,
    max_bytes: int = 1_000_000,
) -> str:
    """Fetch a URL and return its contents as text.

    This is intended for lightweight research/document retrieval.

    Safety/ergonomics features:
    - Only allows http/https URLs
    - Best-effort SSRF protection (blocks localhost/private IP ranges)
    - Content-type gate (refuses obvious binary types)
    - Byte cap + text truncation to protect the context window

    Args:
        url: The URL to fetch.
        max_chars: Max characters returned to the model (tool output truncation).
        timeout_s: Network timeout in seconds.
        max_bytes: Max bytes downloaded from the response body.

    Returns:
        A string containing basic fetch metadata + the fetched text (possibly truncated).
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return "Error: only http/https URLs are supported."
    if not parsed.netloc:
        return "Error: URL must include a hostname."

    host = parsed.hostname or ""
    blocked, reason = _host_looks_local_or_private(host)
    if blocked:
        return f"Error: blocked URL host {host!r}: {reason}."

    headers = {
        "User-Agent": "pydantask-fetch/0.1 (+https://github.com/pydantic/pydantask)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,application/json;q=0.7,*/*;q=0.1",
    }

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=timeout_s
        ) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                status = resp.status_code
                final_url = str(resp.url)
                content_type = (resp.headers.get("content-type") or "").lower()

                # Refuse obvious binary content.
                allowed = any(
                    ct in content_type
                    for ct in (
                        "text/",
                        "application/json",
                        "application/xml",
                        "application/xhtml+xml",
                        "application/javascript",
                    )
                )
                if not allowed:
                    return (
                        f"Error: unsupported content-type {content_type!r} for {final_url}. "
                        "This tool only returns text-like responses."
                    )

                resp.raise_for_status()

                buf = bytearray()
                truncated_bytes = False
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    remaining = max_bytes - len(buf)
                    if remaining <= 0:
                        truncated_bytes = True
                        break
                    if len(chunk) > remaining:
                        buf.extend(chunk[:remaining])
                        truncated_bytes = True
                        break
                    buf.extend(chunk)

                encoding = resp.encoding or "utf-8"
                text = buf.decode(encoding, errors="replace")

                if truncated_bytes:
                    text += "\n\n...[TRUNCATED: response exceeded max_bytes limit]..."

                text = _truncate_text(text, max_chars=max_chars)

                return (
                    f"Fetched: {final_url}\n"
                    f"Status: {status}\n"
                    f"Content-Type: {content_type or '<unknown>'}\n\n"
                    f"{text}"
                )

    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} while fetching {url!r}."
    except httpx.RequestError as e:
        return f"Error: request failed while fetching {url!r}: {str(e)}"
    except Exception as e:
        return f"Error: unexpected failure while fetching {url!r}: {str(e)}"
