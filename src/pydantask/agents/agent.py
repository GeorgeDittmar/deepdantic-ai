# from asyncio import tasks
import json
import os
import asyncio
import re
from httpx import AsyncClient, HTTPStatusError
from tenacity import (
    wait_exponential,
    retry_if_exception_type,
    stop_after_attempt,
)

import uuid
from collections import Counter

from loguru import logger

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from typing import List, Optional, Literal, Any, Dict, Callable, Union
from datetime import datetime
from asyncio import TaskGroup
from pydantask.capabilities.introspection import (
    callable_input_schema,
    format_callable_inputs_for_prompt,
    unwrap_callable,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.usage import UsageLimits

from pydantask.capabilities.runner_v2 import as_runner, CapabilityRunner
from pydantask.agents.scheduler import Scheduler
from pydantask.agents.supervisor_tools import SupervisorTools
from pydantask.agents.executor import TaskExecutor
from pydantask.agents.critic_handler import CriticHandler
from pathlib import Path
from pydantic_ai.models import Model
from pydantask.prompts.prompts_v2 import (
    SUPERVISOR_INPUT_PROMPT,
    WORKER_AGENT_SYS_PROMPT,
    BOOTSTRAP_INSTRUCT,
    ORCHESTRATION_INSTRUCT,
    COMPRESSED_RESEARCH_SYS_PROMPT,
    COMPRESSED_SUPER_PROMPT,
    COMPRESSED_CRITIC_SYS_PROMPT,
    COMPRESSED_WORKER_SYS_PROMPT,
    COMPRESSED_PRODUCER_SYS_PROMPT,
)

from pydantask.models import (
    RuntimeState,
    TaskItem,
    Plan,
    TaskQAResult,
    TaskStatus,
    SupervisorDecision,
    CapabilityDescription,
    TaskResult,
    ArtifactRef,
    PydanTaskRunResult,
    TaskRunDeps,
    TracingBackend,
)

# Default tool wiring is intentionally in-memory focused.
# Filesystem tools still exist in `pydantask.tools.default_tools` but are not enabled by default.
from pydantask.tools.default_tools import (
    append_scratch_note,
    fetch_url_content,
    get_current_datetime,
    get_task_result,
    list_completed_tasks,
    read_scratch_notes,
    think_tool,
)
from pydantask.tools.artifact_tools import (
    put_artifact,
    get_artifact,
    list_artifacts,
    attach_artifact_to_result,
    store_file_as_artifact,
)

from pydantask.manager.checkpointer import CheckpointEvent, CheckpointRecorder
from pydantask.observe.tracing import (
    traced,
    init_tracing_backend,
    autodetect_tracing_backend,
    flush_tracing,
)
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

EVENT_RESULT_DETAIL_TRUNCATION = 4_000
# When a task result is too large to keep inline in the event log, we persist
# the full JSON payload under the checkpoint directory and store only a pointer
# (plus a truncated preview) in events.jsonl.
TASK_RESULT_ARTIFACT_DIRNAME = "task_results"

# Consult runs are intended to be quick and cheap.
CONSULT_TOTAL_TOKENS_LIMIT = 1_200

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


class DeepAgent:
    """Pydantic AI based DeepAgent that manages sub-agents to achieve complex goals."""

    def __init__(
        self,
        objective: str,
        model: str | Model = "gpt-5.2",
        # seed_plan: Plan | None = None,
        # planning_mode: Literal["llm", "fixed", "hybrid"] = "llm",
        default_capabilities_enabled: bool = False,
        custom_supervisor: Agent | None = None,
        max_steps: int = 20,
        max_steps_no_progress: int = 5,
        max_concurrent_tasks: int = 4,
        set_token_budget: Union[int, None] = None,
        capabilities: Union[None, list[CapabilityDescription]] = None,
        # default output type for the producer agent, can be set to a default type or custom pydantic model for better structure and validation of final output
        # output_type: Type = TaskResult,
        # planning_mode: str = "dynamic",  # "static" | "dynamic"
        trace: bool = False,
        checkpoint: bool = False,
        checkpoint_dir: Path | str | None = None,
        resume_from_checkpoint: bool = False,
        verbose_logging: bool = False,
    ):
        """Initialize a DeepAgent instance.

        Args:
            objective: The overall objective / task the deep agent is working on.
            model: Model identifier or ``pydantic_ai.models.Model`` instance to use
                for all sub-agents. Defaults to ``"gpt-5.2"``.
            default_capabilities_enabled: If ``True``, register built-in capabilities
                (research, worker, producer) by default.
            max_steps: Maximum number of DeepAgent control-loop iterations to run
                before forcing termination.
            max_steps_no_progress: Number of consecutive cycles with no executed tasks
                before aborting with a deadlock report.
            max_concurrent_tasks: Maximum number of tasks to run concurrently in each
                control-loop iteration. Tasks exceeding this limit are batched and
                executed sequentially in chunks. Defaults to ``4``.
            set_token_budget: Optional global token budget for the run.
            capabilities: Additional ``CapabilityDescription`` objects to register as
                callable sub-agents alongside the built-ins.
            trace: If ``True``, auto-configure tracing via the configured backend.
            checkpoint: If ``True``, enable event-sourced checkpoint logging for recovery.
            checkpoint_dir: Optional directory to reuse for checkpoints when resuming a run.
                If omitted, a unique directory under ``_checkpoint/`` is created.
            resume_from_checkpoint: If ``True``, attempt to replay from an existing
                checkpoint when starting the agent.
            verbose_logging: If ``True``, log richer debugging information during
                execution.
        """

        if trace:
            init_tracing_backend(autodetect_tracing_backend())

        # `model` can be either:
        #   - a pydantic_ai Model instance (fully custom)
        #   - a bare model name (defaults to OpenAI), e.g. "gpt-4.1-mini"
        #   - a provider-prefixed string, e.g. "openai:gpt-4.1-mini" or "anthropic:claude-sonnet-4-5"
        self.model_name: str = (
            model if isinstance(model, str) else model.__class__.__name__
        )

        if objective is None:
            raise TypeError("DeepAgent requires 'objective' to be provided")

        # if planning_mode in {"fixed", "hybrid"} and seed_plan is None:
        #     raise ValueError(
        #         "seed_plan must be provided when planning_mode is 'fixed' or 'hybrid'"
        #     )

        self.objective: str = objective
        self._max_steps: int = max_steps  # Max steps to prevent infinite loops
        self.max_concurrent_tasks: int = max_concurrent_tasks
        self.token_budget: Union[int, None] = set_token_budget
        self.verbose = verbose_logging
        # self.output_type = output_type
        self.planning_mode = ""
        self.seed_plan: Union[Plan, None] = None
        self._retry_client = self._create_retrying_client()

        # Checkpointing / resume semantics:
        # - `checkpoint=True` enables writing events.
        # - `checkpoint_dir=...` forces checkpointing on and chooses the directory.
        # - `resume=True` requires `checkpoint_dir` and will replay
        #   events from that directory on `run()`.
        if resume_from_checkpoint and checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be provided when resume=True")

        if checkpoint_dir is not None or checkpoint:
            checkpoint = True

        self.checkpoint = checkpoint
        self.resume = resume_from_checkpoint

        # Concurrency guardrails:
        # - `_plan_lock` protects plan-level mutations and task claiming (READY->RUNNING).
        self._plan_lock = asyncio.Lock()

        self.checkpoint_path: Path | None = None
        self._checkpoint_recorder: CheckpointRecorder | None = None
        if self.checkpoint:
            self.checkpoint_path = (
                Path(checkpoint_dir)
                if checkpoint_dir is not None
                else Path(f"_checkpoint/{uuid.uuid4()}/")
            )
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
            self._checkpoint_recorder = CheckpointRecorder(self.checkpoint_path)

        # Supervisor tools (extracted from this class for testability).
        self._supervisor_tools = SupervisorTools(
            context_resolver=lambda ctx: ctx.deps,
        )

        # Scheduler (extracted from this class for testability).
        self._scheduler = Scheduler(
            context_resolver=lambda ctx: ctx,
        )

        # Build the shared model used by all sub-agents.
        # TODO: Future state allow for configuration of what models to use per capability
        # We inject the retrying httpx client into the provider for durability.
        self._retry_model = self._build_model(model)

        if custom_supervisor:
            self._supervisor_agent = custom_supervisor

        self._supervisor_agent = custom_supervisor or Agent(
            model=self._retry_model,
            name="_dynamic_Supervisor_Agent",
            system_prompt=COMPRESSED_SUPER_PROMPT,
            output_type=SupervisorDecision,
            deps_type=RuntimeState,
            tools=self._default_supervisor_tools(),
            end_strategy="exhaustive",
        )

        _default_capabiliites = []
        if default_capabilities_enabled:
            _default_capabiliites = self._setup_default_capabilities()

        self._capability_registry = self._setup_capability_registry(
            _default_capabiliites, additonal_capabilities=capabilities
        )

        # Task executor (extracted for testability).
        self._executor = TaskExecutor(
            context_resolver=lambda ctx: ctx,
            capability_registry=self._capability_registry,
            record_event_cb=self._record_event,
            record_task_status_event_cb=self._record_task_status_event,
            record_metadata_append_cb=self._record_metadata_append,
            coerce_output_cb=self._coerce_output_to_task_result,
            record_task_result_cb=self._record_task_result,
        )
        # Set objective for prompt construction (executor falls back to this).
        self._executor.objective = self.objective
        # Wire cascade_cancellations callback (scheduler method).
        self._executor._cascade_cancellations_cb = self._scheduler.cascade_cancellations
        # Wire run_task_cb using a lambda so test mocks on da.execute are picked up.
        self._executor._run_task_cb = (
            lambda capability, step, ctx: self.execute(capability, step, ctx)
        )

        # Critic handler (extracted for testability).
        self._critic_handler = CriticHandler(
            record_event_cb=self._record_event,
            record_task_status_event_cb=self._record_task_status_event,
        )

        self._critic_agent = Agent(
            model=self._retry_model,
            name="_default_Critic_Agent",
            system_prompt=COMPRESSED_CRITIC_SYS_PROMPT,
            output_type=TaskQAResult,
            deps_type=RuntimeState,
            tools=[
                get_current_datetime,
                think_tool,
                # Evidence retrieval (bounded)
                get_task_result,
                list_artifacts,
                get_artifact,
            ],
            # end_strategy="exhaustive",
        )
        # Scheduler/system notes injected into the next supervisor prompt.
        self._last_scheduler_report: str = ""

    async def _coerce_output_to_task_result(
        self,
        step: TaskItem,
        output: Any,
        *,
        runtime_state: RuntimeState | None = None,
    ) -> TaskResult:
        """Coerce arbitrary capability outputs into the canonical TaskResult.

        Design goal: capability authors can return "anything" (str/dict/Pydantic model),
        and DeepAgent will normalize it for downstream evaluation/synthesis.

        Enterprise/ops goal: if the output references on-disk files, ingest them into
        the run's artifact store so the critic/producer can retrieve contents via
        `get_artifact` (rather than relying on host paths).
        """

        rejected_files: list[dict[str, Any]] = []

        def _allowed_source_roots() -> list[Path]:
            """Roots from which host-side files may be ingested as artifacts.

            IMPORTANT SECURITY NOTE:
            We do NOT want to ingest arbitrary host files just because a model
            mentions a path (e.g. "/etc/passwd"). We therefore restrict source
            file ingestion to a small allowlist of safe roots.

            Current policy:
              - allow files under ./tmp
              - allow files under the active checkpoint directory (if available)
            """
            roots: list[Path] = []

            # Common deterministic tools write here.
            roots.append((Path.cwd() / "tmp").resolve())

            # If checkpointing is enabled, allow ingesting files created under it.
            if runtime_state is not None:
                recorder = getattr(runtime_state, "checkpoint_recorder", None)
                directory = getattr(recorder, "directory", None)
                if isinstance(directory, Path):
                    roots.append(directory.resolve())

            # Dedupe
            out: list[Path] = []
            seen: set[str] = set()
            for r in roots:
                key = str(r)
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
            return out

        def _is_allowed_source_file(p: Path) -> bool:
            try:
                rp = p.resolve()
            except Exception:
                return False

            for root in _allowed_source_roots():
                try:
                    if rp == root or root in rp.parents:
                        return True
                except Exception:
                    continue
            return False

        def _existing_files_from_text(text: str) -> list[Path]:
            """Extract safe, existing files from text.

            Only returns files under the allowlisted roots.
            """
            candidates: list[Path] = []

            # 1) If the entire output is a path, prefer that.
            p = Path(text)
            if p.exists() and p.is_file():
                if _is_allowed_source_file(p):
                    candidates.append(p)
                else:
                    rejected_files.append(
                        {"path": str(p), "reason": "outside_allowed_roots"}
                    )

            # 2) Otherwise, attempt to extract path-like tokens.
            # Best-effort: only keep tokens that resolve to an existing file AND
            # live under allowlisted roots.
            for token in re.findall(r"[^\s'\"]+", text):
                cleaned = token.strip().strip(".,;:()[]{}<>")
                if not cleaned:
                    continue
                cp = Path(cleaned)
                if not (cp.exists() and cp.is_file()):
                    continue
                if _is_allowed_source_file(cp):
                    candidates.append(cp)
                else:
                    rejected_files.append(
                        {"path": str(cp), "reason": "outside_allowed_roots"}
                    )

            # Dedupe while preserving order.
            out: list[Path] = []
            seen: set[str] = set()
            for c in candidates:
                try:
                    key = str(c.resolve())
                except Exception:
                    key = str(c)
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
            return out

        if isinstance(output, TaskResult):
            if output.task_id != step.task_id:
                output.task_id = step.task_id
            return output

        meta: dict[str, Any] = {
            "raw_output_type": type(output).__name__,
            "coerced": True,
        }

        if output is None:
            return TaskResult(
                task_id=step.task_id,
                status=TaskStatus.ERRORED,
                summary="Capability returned no output.",
                detailed_output="",
                error_msg="Capability returned None",
                metadata=meta,
            )

        if isinstance(output, BaseModel):
            try:
                dumped = output.model_dump(mode="json")
                text = output.model_dump_json(indent=2)
            except Exception:
                dumped = {}
                text = str(output)

            return TaskResult(
                task_id=step.task_id,
                status=TaskStatus.COMPLETED,
                summary=f"Capability returned {type(output).__name__}.",
                detailed_output=text,
                data=dumped if isinstance(dumped, dict) else {},
                metadata=meta,
            )

        if isinstance(output, (dict, list)):
            try:
                text = json.dumps(output, ensure_ascii=False, indent=2)
            except Exception:
                text = str(output)

            data_payload: dict[str, Any] = {}
            if isinstance(output, dict):
                if len(text) <= 16_000:
                    data_payload = output
                else:
                    meta["data_omitted_reason"] = "too_large"

            return TaskResult(
                task_id=step.task_id,
                status=TaskStatus.COMPLETED,
                summary="Capability returned structured data.",
                detailed_output=text,
                data=data_payload,
                metadata=meta,
            )

        # Normalize to string and attempt file-ingestion.
        text = str(output)
        files: list[Path] = []
        if isinstance(output, Path):
            if output.exists() and output.is_file():
                files = [output]
        elif (
            isinstance(output, (list, tuple))
            and output
            and all(isinstance(x, (str, Path)) for x in output)
        ):
            # Allow callables to return ["tmp/a.txt", "tmp/b.txt"]
            for item in output:
                if isinstance(item, Path):
                    if item.exists() and item.is_file():
                        files.append(item)
                else:
                    files.extend(_existing_files_from_text(str(item)))
        elif isinstance(output, str):
            files = _existing_files_from_text(output)

        artifact_refs: list[ArtifactRef] = []
        file_records: list[dict[str, Any]] = []
        if runtime_state is not None and files:
            for p in files:
                try:
                    ref = await store_file_as_artifact(
                        runtime_state,
                        file_path=p,
                        task_id=step.task_id,
                        name=p.name,
                    )
                    artifact_refs.append(ref)
                    file_records.append(
                        {
                            "path": str(p),
                            "artifact_id": ref.artifact_id,
                            "uri": ref.uri,
                            "mime_type": ref.mime_type,
                            "size_bytes": ref.size_bytes,
                        }
                    )
                except Exception as e:
                    file_records.append({"path": str(p), "error": str(e)})

        if file_records:
            meta["file_outputs"] = file_records
        if rejected_files:
            meta["file_outputs_rejected"] = rejected_files

        # Summary/preview.
        summary = (text.strip().splitlines()[0] if text else "").strip()
        if len(summary) > 280:
            summary = summary[:280] + "..."

        detail_parts: list[str] = [text]
        if artifact_refs:
            detail_parts.append(
                "\n\n---\nFile outputs ingested as artifacts (preferred for review):"
            )
            for r in artifact_refs:
                detail_parts.append(
                    f"- {r.name or '<unnamed>'}: artifact_id={r.artifact_id} uri={r.uri}"
                )
                if r.preview:
                    detail_parts.append(
                        "  preview:\n" + self._truncate_text(r.preview, 1500)
                    )

        tr = TaskResult(
            task_id=step.task_id,
            status=TaskStatus.COMPLETED,
            summary=(
                f"Produced {len(artifact_refs)} artifact file(s)."
                if artifact_refs
                else (summary or "Capability returned text output.")
            ),
            detailed_output="\n".join(detail_parts).strip(),
            metadata=meta,
        )
        if artifact_refs:
            tr.artifacts.extend(artifact_refs)
        return tr

    def _setup_default_capabilities(self) -> List[CapabilityDescription]:

        # NOTE: Filesystem tools exist in `pydantask.tools.default_tools`, but are not
        # enabled by default. The harness is currently in-memory focused.

        # TODO: rework some of these tools
        tavily_api_key = os.getenv("TAVILY_API_KEY", None)

        _default_research_tool_set = [
            think_tool,
            append_scratch_note,
            read_scratch_notes,
            get_current_datetime,
            # Artifact store (segregated, resumable)
            put_artifact,
            get_artifact,
            list_artifacts,
            attach_artifact_to_result,
            # fetch_url_content,
            # Cross-agent "consult" (bounded, logged)
            self.consult_capability,
        ]

        if not tavily_api_key:
            logger.info(
                "Tavily api key not found. Defaulting to built in Duck Duck Go search tool."
            )
            _default_research_tool_set.append(duckduckgo_search_tool())
        else:
            _default_research_tool_set.append(tavily_search_tool(tavily_api_key))

        self._researcher_agent = Agent(
            model=self._retry_model,
            name="_default_Research_Agent",
            system_prompt=COMPRESSED_RESEARCH_SYS_PROMPT,
            tools=_default_research_tool_set,
            deps_type=TaskRunDeps,
            output_type=TaskResult,
        )

        general_worker_agent = Agent(
            model=self._retry_model,
            name="_default_General_Worker_Agent",
            system_prompt=COMPRESSED_WORKER_SYS_PROMPT,
            deps_type=TaskRunDeps,
            output_type=TaskResult,
            tools=[
                # list_documents,
                list_completed_tasks,
                get_task_result,
                # Artifact store (segregated, resumable)
                put_artifact,
                get_artifact,
                list_artifacts,
                attach_artifact_to_result,
                # Cross-agent "consult" (bounded, logged)
                think_tool,
                append_scratch_note,
                read_scratch_notes,
                get_current_datetime,
            ],
        )

        producer_agent = Agent(
            model=self._retry_model,
            name="_default_Producer_agent",
            system_prompt=COMPRESSED_PRODUCER_SYS_PROMPT,
            deps_type=TaskRunDeps,
            output_type=TaskResult,
            tools=[
                # Plan / history inspection
                list_completed_tasks,
                get_task_result,
                # Artifact store (segregated, resumable)
                put_artifact,
                get_artifact,
                list_artifacts,
                attach_artifact_to_result,
                # Cross-agent "consult" (bounded, logged)
                # Reflection
                think_tool,
            ],
        )

        producer = CapabilityDescription(
            name="producer_agent",
            description="Produces output based on information from various sources and sub agents.",
            tool_func=as_runner(producer_agent),
        )

        researcher = CapabilityDescription(
            name="research_agent",
            description="Tool to research information. This could include searching the web or querying a data source.",
            tool_func=as_runner(self._researcher_agent),
        )

        gen_worker = CapabilityDescription(
            name="worker_agent",
            description=(
                "General-purpose worker for analysis, summarization, document editing, "
                "code or log interpretation, and other non-research tasks that operate on "
                "existing context."
            ),
            tool_func=as_runner(general_worker_agent),
        )

        capabilities_list = [producer, researcher, gen_worker]

        return capabilities_list

    async def aclose(self) -> None:
        """Close underlying resources used by this ``DeepAgent`` instance.

        This is primarily responsible for flushing any tracing backends and
        closing the shared async HTTP client used by the model providers.
        Safe to call multiple times.
        """
        try:
            # best-effort; safe to call even if tracing is disabled
            flush_tracing()
        finally:
            if getattr(self, "_retry_client", None) is not None:
                await self._retry_client.aclose()

    async def __aenter__(self) -> "DeepAgent":
        """Enter the async context manager and return this ``DeepAgent``.

        Allows ``async with DeepAgent(...) as agent: ...`` usage.
        """
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the async context manager, ensuring resources are cleaned up."""
        await self.aclose()

    def _default_supervisor_tools(self) -> list[Callable]:
        """Return the set of tools exposed to the supervisor agent.

        The selected toolset depends on :attr:`planning_mode`:

        * ``fixed``: supervisor can update/cancel tasks and view QA reports, but
          cannot add or patch tasks.
        * ``llm`` / ``hybrid``: supervisor can also add and patch tasks.

        Note: This is enforced by tool registration (not just prompting), so in
        ``fixed`` mode the supervisor LLM cannot call ``add_task``/``patch_task``.
        """
        base_tools = [
            self._supervisor_tools.update_task_status,
            self._supervisor_tools.cancel_task,
            self._supervisor_tools.view_qa_report,
            get_current_datetime,
            think_tool,
        ]

        mutating_tools = [
            self._supervisor_tools.add_task,
            self._supervisor_tools.patch_task,
            self._supervisor_tools.mark_final_task,
        ]

        if self.planning_mode == "fixed":
            return base_tools

        return base_tools + mutating_tools

    def _build_model(self, model: str | Model) -> Model:
        """Construct a ``pydantic_ai`` model, wiring in the shared HTTP client.

        The ``model`` parameter may be either:

        * A bare model name (e.g. ``"gpt-4.1-mini"``) which defaults to the
          OpenAI provider.
        * A provider-prefixed string such as ``"openai:gpt-4.1-mini"`` or
          ``"anthropic:claude-sonnet-4-5"``.
        * An already-instantiated ``pydantic_ai.models.Model`` instance, which is
          returned unchanged.
        """
        if isinstance(model, Model):
            return model

        provider_name: str
        model_name: str
        if ":" in model:
            provider_name, model_name = model.split(":", 1)
            provider_name = provider_name.strip().lower()
            model_name = model_name.strip()
        else:
            provider_name, model_name = "openai", model

        if provider_name in {"openai", "openai_compat", "openrouter"}:
            # NOTE: "openrouter" here assumes OpenAI-compatible API. If you want true
            # OpenRouter defaults (headers/routing), we may want OpenRouterProvider. Dunno
            return OpenAIChatModel(
                model_name, provider=OpenAIProvider(http_client=self._retry_client)
            )

        if provider_name == "anthropic":
            return AnthropicModel(
                model_name,
                provider=AnthropicProvider(http_client=self._retry_client),
            )

        raise ValueError(
            f"Unsupported model provider prefix: {provider_name!r}. "
            "Use e.g. 'openai:...' or 'anthropic:...' or pass a Model instance."
        )

    def _create_retrying_client(self):
        """Create an ``httpx.AsyncClient`` with robust retry behaviour.

        The returned client uses ``AsyncTenacityTransport`` with sensible
        defaults for rate limits and transient network failures. See
        https://ai.pydantic.dev/retries/ for more details.
        """

        def should_retry_status(response):
            """Raise exceptions for retryable HTTP status codes."""
            if response.status_code in (429, 502, 503, 504):
                response.raise_for_status()  # This will raise HTTPStatusError

        transport = AsyncTenacityTransport(
            config=RetryConfig(
                # Retry on HTTP errors and connection issues
                retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
                # Smart waiting: respects Retry-After headers, falls back to exponential backoff
                wait=wait_retry_after(
                    fallback_strategy=wait_exponential(multiplier=1, max=60),
                    max_wait=300,
                ),
                # TODO: make this configurable
                stop=stop_after_attempt(3),
                # Re-raise the last exception if all retries fail
                reraise=True,
            ),
            validate_response=should_retry_status,
        )

        return AsyncClient(transport=transport)

    def _setup_capability_registry(
        self,
        default_capabilities,
        additonal_capabilities: Union[None, list[CapabilityDescription]] = None,
    ) -> Dict:
        """Create the default sub-agent capability registry.

        This wires up the built-in producer, researcher, and general worker
        agents, and optionally merges any extra ``CapabilityDescription``
        instances supplied by the caller.

        Args:
            additonal_capabilities: Additional capabilities to register on top of
                the built-in sub-agents.

        Returns:
            Dict[str, CapabilityDescription]: Mapping from capability name to
            its description and callable agent/tool.
        """

        _capabilities_list = default_capabilities
        # if additional sub agents been supplied then add those to the registry
        if additonal_capabilities:
            _capabilities_list.extend(additonal_capabilities)

        _capability_registry = {
            capability.name: capability for capability in _capabilities_list
        }

        # each agent gets its own unique id
        return _capability_registry

    def _initialize_runtime_state(self, objective: str, registry: dict) -> RuntimeState:
        """Create the initial :class:`RuntimeState` for a new DeepAgent run.

        This initializes an empty plan. If ``seed_plan`` was provided when the
        DeepAgent was constructed, it is applied at the start of :meth:`run`.

        Args:
            objective: Top-level objective for this DeepAgent execution.
            registry: Mapping of capability names to ``CapabilityDescription``
                instances.

        Returns:
            A freshly initialized ``RuntimeState`` with an empty plan and
            ``next_task_id`` set to ``1``.
        """
        runtime_state = RuntimeState(
            objective=objective, capability_registry=registry, next_task_id=1
        )
        runtime_state.checkpoint_recorder = self._checkpoint_recorder
        return runtime_state

    async def _checkpoint_state(self, runtime: RuntimeState):
        """Persist a lightweight runtime summary when checkpointing is enabled."""
        if not self._checkpoint_recorder:
            return

        summary = {
            "ts": datetime.now().isoformat(),
            "runtime_steps": runtime.runtime_steps,
            "total_tasks": len(runtime.plan),
            "status_counts": dict(
                Counter(t.status.value for t in runtime.plan.values())
            ),
            "next_task_id": runtime.next_task_id,
        }
        await self._checkpoint_recorder.record_summary(summary)

    async def _replay_checkpoint(self, runtime_state: RuntimeState) -> None:
        if not self._checkpoint_recorder:
            return

        events = await self._checkpoint_recorder.load_events()
        if not events:
            return

        for event in events:
            self._apply_event(runtime_state, event)

        if runtime_state.plan:
            max_existing = max(runtime_state.plan.keys()) + 1
            runtime_state.next_task_id = max(runtime_state.next_task_id, max_existing)

    def _apply_event(self, runtime_state: RuntimeState, event: CheckpointEvent) -> None:
        payload = event.payload or {}
        event_type = event.type

        if event_type == "task_added":
            task_data = payload.get("task")
            if not task_data:
                return
            task = TaskItem(**task_data)
            runtime_state.plan[task.task_id] = task
            runtime_state.next_task_id = max(
                runtime_state.next_task_id,
                payload.get("next_task_id", task.task_id + 1),
            )
            return

        if event_type == "task_patched":
            task_id = payload.get("task_id")
            if task_id is None or task_id not in runtime_state.plan:
                return
            task = runtime_state.plan[task_id]

            if "sub_task_objective" in payload:
                task.sub_task_objective = payload["sub_task_objective"]
            if "dependencies" in payload:
                task.sub_task_dependencies = payload["dependencies"]
            if "capability" in payload:
                task.capability = payload["capability"]
            if "parameters" in payload and isinstance(payload["parameters"], dict):
                # Patch semantics for parameters are merge/update.
                if not isinstance(getattr(task, "parameters", None), dict):
                    task.parameters = {}
                task.parameters.update(payload["parameters"])
            if "is_final" in payload:
                task.is_final = bool(payload["is_final"])
            return

        if event_type == "final_task_set":
            task_id = payload.get("task_id")
            if task_id is None:
                return

            # Enforce the invariant: at most one task is marked final.
            for t in runtime_state.plan.values():
                t.is_final = False

            if task_id in runtime_state.plan:
                runtime_state.plan[task_id].is_final = True
            return

        if event_type == "task_status_updated":
            task_id = payload.get("task_id")
            if task_id is None or task_id not in runtime_state.plan:
                return
            task = runtime_state.plan[task_id]
            status_value = payload.get("status")
            if status_value is not None:
                task.status = TaskStatus(status_value)
            if "error_msg" in payload:
                task.error_msg = payload.get("error_msg")
            reason = payload.get("reason")
            if reason:
                history = task.metadata.setdefault("status_history", [])
                if isinstance(history, list):
                    history.append(
                        {
                            "ts": event.ts.isoformat(),
                            "status": task.status.value,
                            "reason": reason,
                        }
                    )
            return

        if event_type == "task_result":
            task_id = payload.get("task_id")
            result_payload = payload.get("result")
            if (
                task_id is None
                or result_payload is None
                or task_id not in runtime_state.plan
            ):
                return

            # If the event references a sidecar file, prefer that full payload.
            full_path = payload.get("full_result_path")
            if isinstance(full_path, str) and full_path:
                loaded = self._load_full_task_result_payload(full_path)
                if loaded is not None:
                    result_payload = loaded

            runtime_state.plan[task_id].result = TaskResult(**result_payload)
            return

        if event_type == "task_metadata_appended":
            task_id = payload.get("task_id")
            key = payload.get("key")
            value = payload.get("value")
            if task_id is None or key is None or task_id not in runtime_state.plan:
                return
            task = runtime_state.plan[task_id]
            existing = task.metadata.get(key)
            if existing is None:
                task.metadata[key] = value
            elif isinstance(existing, list):
                existing.append(value)
            elif isinstance(existing, str):
                task.metadata[key] = existing + f"\n\n{value}"
            else:
                task.metadata[key] = value
            return

        if event_type == "scratch_note_appended":
            task_id = payload.get("task_id")
            if task_id is None or task_id not in runtime_state.plan:
                return
            note = payload.get("note", "")
            key = "scratch_notes"
            existing = runtime_state.plan[task_id].metadata.get(key, "")
            runtime_state.plan[task_id].metadata[key] = existing + f"\n\n{note}"
            return

        if event_type == "critic_feedback":
            task_id = payload.get("task_id")
            if task_id is None or task_id not in runtime_state.plan:
                return
            feedback_payload = payload.get("feedback")
            if feedback_payload is not None:
                runtime_state.plan[task_id].task_feedback = TaskQAResult(
                    **feedback_payload
                )
            if "attempt_count" in payload:
                runtime_state.plan[task_id].attempt_count = payload["attempt_count"]
            return

        # supervisor_decision and other audit events do not mutate state on replay.

    async def _record_event(
        self, event_type: CheckpointEventType, payload: Dict[str, Any]
    ) -> None:
        if self._checkpoint_recorder:
            await self._checkpoint_recorder.record(event_type, payload)

    async def _record_task_status_event(
        self,
        task_id: int,
        status: TaskStatus,
        *,
        reason: str | None = None,
        error_msg: str | None = None,
    ) -> None:
        payload: Dict[str, Any] = {"task_id": task_id, "status": status.value}
        if reason:
            payload["reason"] = reason
        if error_msg:
            payload["error_msg"] = error_msg
        await self._record_event("task_status_updated", payload)

    def _persist_full_task_result_payload(
        self, task_id: int, result_payload: Dict[str, Any]
    ) -> str | None:
        """Persist the full TaskResult payload under the checkpoint directory.

        Returns a *relative* path (from checkpoint root) that can be stored in
        the event log, or ``None`` if persistence is unavailable.
        """
        if self.checkpoint_path is None:
            return None

        artifacts_dir = self.checkpoint_path / TASK_RESULT_ARTIFACT_DIRNAME
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        relpath = f"{TASK_RESULT_ARTIFACT_DIRNAME}/task_{task_id}.json"
        path = self.checkpoint_path / relpath
        with path.open("w", encoding="utf-8") as fh:
            json.dump(result_payload, fh, ensure_ascii=False)

        return relpath

    def _load_full_task_result_payload(self, relpath: str) -> Dict[str, Any] | None:
        """Load a full TaskResult payload previously persisted by this agent."""
        if self.checkpoint_path is None:
            return None

        path = self.checkpoint_path / relpath
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception:
            return None

        return None

    async def _record_task_result(self, task: TaskItem) -> None:
        if not self._checkpoint_recorder or not task.result:
            return

        # Use JSON mode so datetimes (e.g. SourceRef.accessed_at) are serializable.
        result_payload: Dict[str, Any] = task.result.model_dump(mode="json")

        full_result_path: str | None = None
        detailed_output = result_payload.get("detailed_output") or ""
        if detailed_output and len(detailed_output) > EVENT_RESULT_DETAIL_TRUNCATION:
            # Persist the full payload to a sidecar file so replay can restore it.
            full_result_path = await asyncio.to_thread(
                self._persist_full_task_result_payload, task.task_id, result_payload
            )

            truncation_notice = f"\n\n...[TRUNCATED {len(detailed_output) - EVENT_RESULT_DETAIL_TRUNCATION} chars]..."
            result_payload["detailed_output"] = (
                detailed_output[:EVENT_RESULT_DETAIL_TRUNCATION] + truncation_notice
            )

        payload: Dict[str, Any] = {"task_id": task.task_id, "result": result_payload}
        if full_result_path:
            payload["full_result_path"] = full_result_path

        await self._record_event("task_result", payload)

    async def _record_metadata_append(self, task_id: int, key: str, value: Any) -> None:
        if not self._checkpoint_recorder:
            return
        await self._record_event(
            "task_metadata_appended", {"task_id": task_id, "key": key, "value": value}
        )

    def _format_capabilities(self) -> str:
        """Format all registered capabilities into a planner-friendly string.

        Each line is of the form: ``- <capability_name>: <description>``.

        For callable capabilities (non-Agent), we also display an inferred input
        contract (argument names/types) so the supervisor knows what structured
        values to include when creating tasks.
        """
        lines: list[str] = []
        for name, desc in self._capability_registry.items():
            description = getattr(desc, "description", "")
            lines.append(f"- {name}: {description}")

            tool_func = getattr(desc, "tool_func", None)
            func = unwrap_callable(tool_func)
            if func is None:
                continue

            schema = callable_input_schema(func)
            if (schema.get("required") or []) or (schema.get("optional") or []):
                lines.append("  " + format_callable_inputs_for_prompt(schema))
                lines.append("  " + "provide via TaskItem.parameters")

        return "\n".join(lines)

    def _format_plan(self, plan: Plan):
        """Format a :class:`Plan` instance into a human-readable multi-line string."""
        lines = []
        for task in plan.tasks:
            id = task.task_id
            sub_task_obj = task.sub_task_objective
            task_status = task.status
            metadata = task.metadata
            lines.append(
                f"- Task ID:{id}\n sub_task_obj: {sub_task_obj} \n task_status: {task_status}\n metadata: {metadata}"
            )
        return "\n".join(lines)

    def _format_supervisor_input_prompt(self, ctx: RuntimeState) -> str:
        """Build the composite prompt passed to the supervisor agent.

        The prompt includes the overall objective, a summarized status board of
        all tasks in the plan, and a list of available capabilities.
        """
        # Pre-format the plan to ensure the LLM sees a clean "Status Board"
        capability_display = self._format_capabilities()

        plan_display_lines = []
        for t in ctx.plan.values():
            line = (
                f"- Task ID: {t.task_id} | Status: [{t.status.value}] "
                f"| Final: {getattr(t, 'is_final', False)} "
                f"| Objective: {t.sub_task_objective} "
                f"| Dependencies: {t.sub_task_dependencies}"
            )

            fb = getattr(t, "task_feedback", None)
            if fb is not None:
                # Adjust these fields to match TaskQAResult
                # verdict = getattr(fb, "passed", None)
                verdict = getattr(fb, "passed", None)
                summary = getattr(fb, "reasoning", None)

                line += "\n  QA: "
                if verdict is not None:
                    line += f"verdict={verdict} "
                if summary:
                    line += f"\n    summary: {summary}"

            plan_display_lines.append(line)

        plan_display = "\n".join(plan_display_lines)

        prompt = SUPERVISOR_INPUT_PROMPT.format(
            objective=ctx.objective,
            plan_display=plan_display,
            agent_display=capability_display,
            now=datetime.now().isoformat(),
            current_year=datetime.now().year,
        )

        if self._last_scheduler_report:
            prompt += (
                "\n\n### SYSTEM SCHEDULER NOTES (deterministic)\n"
                + self._last_scheduler_report.strip()
            )

        return prompt

    def _format_critic_input_prompt(self, task: TaskItem, ctx: RuntimeState) -> str:
        """Construct the evaluation prompt sent to the critic agent.

        The critic receives the overall objective, the ``TaskItem`` definition
        it should be evaluating, the worker's structured ``TaskResult`` (if any),
        and any relevant in-memory documents from the runtime state.

        Note: this harness is currently in-memory focused; do not assume any
        filesystem persistence.
        """
        # Keep the inline prompt small. The critic should fetch the full TaskResult
        # (and any artifacts) via tools.
        result_summary = "<no result>"
        artifact_hints: list[str] = []
        if task.result is not None:
            result_summary = (task.result.summary or "").strip() or "<empty summary>"
            for a in getattr(task.result, "artifacts", []) or []:
                try:
                    artifact_hints.append(
                        f"- name={a.name or '<unnamed>'} artifact_id={a.artifact_id} uri={a.uri} mime={a.mime_type}"
                    )
                except Exception:
                    continue

        artifacts_inline = "\n".join(artifact_hints) if artifact_hints else "<none>"

        _prompt = f"""
You are the QA/critic.

Overall objective:
{ctx.objective}

Task under review:
- task_id: {task.task_id}
- capability: {task.capability}
- objective: {task.sub_task_objective}

Inline summary (non-authoritative):
{result_summary}

Known artifacts on the TaskResult (may be empty):
{artifacts_inline}

Instructions (IMPORTANT):
1) Call `get_task_result(task_id={task.task_id}, max_chars=12000)` to retrieve the full TaskResult JSON.
2) Call `list_artifacts(task_id={task.task_id})` and then `get_artifact(...)` for any relevant artifacts.
3) Judge pass/fail based on the retrieved evidence.
4) If evidence is missing (e.g. only a raw host filesystem path), fail with clear feedback:
   "Please store the produced content as an artifact and attach it to TaskResult.artifacts".

Return a TaskQAResult.
"""
        return _prompt

    def _is_context_limit_error(self, exc: Exception) -> bool:
        """Delegate to :class:`TaskExecutor.is_context_limit_error`."""
        return self._executor.is_context_limit_error(exc)

    def _build_resume_prompt(self, step: TaskItem, error: Exception) -> str:
        """Delegate to :class:`TaskExecutor.build_resume_prompt`."""
        return self._executor.build_resume_prompt(step, error)

    def _truncate_text(self, text: str, max_chars: int | None) -> str:
        """Delegate to :class:`TaskExecutor.truncate_text`."""
        return self._executor.truncate_text(text, max_chars)

    def _remaining_token_budget(self, runtime_state: RuntimeState) -> int | None:
        """Delegate to :class:`TaskExecutor.remaining_token_budget`."""
        return self._executor.remaining_token_budget(runtime_state)

    def _make_usage_limits(self, **kwargs) -> UsageLimits | None:
        """Delegate to :class:`TaskExecutor.make_usage_limits`."""
        return self._executor.make_usage_limits(**kwargs)

    def _extract_total_tokens(self, run_result: Any) -> int | None:
        """Delegate to :class:`TaskExecutor.extract_total_tokens`."""
        return self._executor.extract_total_tokens(run_result)

    def _accumulate_usage(
        self, runtime_state: RuntimeState, run_result: Any, *, label: str
    ) -> None:
        """Delegate to :class:`TaskExecutor.accumulate_usage`."""
        self._executor.accumulate_usage(runtime_state, run_result, label=label)

    async def consult_capability(
        self,
        ctx: RunContext[TaskRunDeps],
        capability: str,
        question: str,
        task_ids: list[int] | None = None,
        max_chars: int = 3_000,
    ) -> str:
        """Tool: Consult Capability (agent-to-agent, bounded & logged).

        This lets a running sub-agent ask another registered capability a narrow
        question *without* asking the supervisor to create new tasks.

        Args:
            ctx: The current task execution deps (TaskRunDeps).
            capability: Which capability to consult (e.g. "research_agent").
            question: The question to ask.
            task_ids: Optional list of task IDs whose results should be included as context.
                Defaults to the caller task's dependencies.
            max_chars: Max characters returned (and persisted) for the answer.

        Returns:
            A concise string answer from the consulted capability.
        """
        runtime_state = ctx.deps.runtime_state
        caller_task = ctx.deps.task

        cap = (capability or "").strip()
        if not cap:
            return "Error: 'capability' must be a non-empty string."

        if cap not in self._capability_registry:
            known = ", ".join(sorted(self._capability_registry.keys()))
            return (
                f"Error: unknown capability {cap!r}. "
                f"Known capabilities: {known if known else '<none>'}."
            )

        # Build a compact context pack from selected upstream tasks.
        include_ids = (
            task_ids
            if task_ids is not None
            else list(getattr(caller_task, "sub_task_dependencies", []) or [])
        )

        ctx_chunks: list[str] = []
        for tid in include_ids:
            t = runtime_state.plan.get(tid)
            if t is None or t.result is None:
                continue
            summary = (t.result.summary or "").strip()
            detail = (t.result.detailed_output or "").strip()
            if len(detail) > 1_200:
                detail = detail[:1_200] + "\n...[detail truncated]..."

            ctx_chunks.append(
                "\n".join(
                    [
                        f"Task {tid} ({t.capability}) objective: {t.sub_task_objective}",
                        f"summary: {summary}",
                        f"detail: {detail}" if detail else "detail: <none>",
                    ]
                )
            )

        upstream_context = "\n\n".join(ctx_chunks)
        upstream_context = self._truncate_text(upstream_context, max_chars=6_000)

        consult_prompt = f"""
You are being consulted by another agent for a narrow, targeted answer.

Overall objective:
{runtime_state.objective}

Caller task:
- task_id: {caller_task.task_id}
- capability: {caller_task.capability}
- objective: {caller_task.sub_task_objective}

Question:
{question}

Relevant upstream context from completed tasks (may be empty):
{upstream_context if upstream_context.strip() else '<none>'}

Instructions:
- Answer from the provided context only.
- Do NOT call tools.
- Keep it concise and actionable.
- If you cannot answer, respond with: INSUFFICIENT_CONTEXT: <what is missing>.
""".strip()

        consulted = self._capability_registry[cap]
        runner = getattr(consulted, "tool_func", None)
        run_method = getattr(runner, "run", None)
        if run_method is None:
            return f"Error: capability {cap!r} is not runnable (missing .run)."

        # Use a synthetic task for the consulted agent so it doesn't treat this
        # as executing the caller's full TaskItem.
        consult_task = TaskItem(
            task_id=caller_task.task_id,
            overall_objective=runtime_state.objective,
            sub_task_objective=f"CONSULT: {question}",
            status=TaskStatus.RUNNING,
            capability=cap,
            sub_task_dependencies=[],
            metadata={"consult_for_task_id": caller_task.task_id},
        )

        consult_deps = TaskRunDeps(runtime_state=runtime_state, task=consult_task)

        # Hard safety: no tool calls during consults.
        consult_limits = self._make_usage_limits(
            tool_calls_limit=0,
            total_tokens_limit=min(
                CONSULT_TOTAL_TOKENS_LIMIT,
                self._remaining_token_budget(runtime_state)
                or CONSULT_TOTAL_TOKENS_LIMIT,
            ),
        )
        resp = await run_method(
            consult_prompt,
            deps=consult_deps,
            usage_limits=consult_limits,
        )
        self._accumulate_usage(runtime_state, resp, label=f"consult:{cap}")
        output = getattr(resp, "output", resp)

        # Normalize to text.
        answer_text: str
        if isinstance(output, TaskResult):
            answer_text = (output.detailed_output or "").strip() or (
                output.summary or ""
            ).strip()
        elif isinstance(output, BaseModel):
            answer_text = output.model_dump_json(indent=2)
        else:
            answer_text = str(output)

        answer_text = self._truncate_text(answer_text, max_chars=max_chars)

        entry = {
            "ts": datetime.now().isoformat(),
            "to": cap,
            "question": self._truncate_text(question, max_chars=1_500),
            "answer": answer_text,
            "task_ids": include_ids,
        }

        caller_task.metadata.setdefault("consultations", [])
        if isinstance(caller_task.metadata.get("consultations"), list):
            caller_task.metadata["consultations"].append(entry)
        else:
            caller_task.metadata["consultations"] = [entry]

        # Persist as an event so checkpoint replay reconstructs it.
        await self._record_metadata_append(caller_task.task_id, "consultations", entry)

        return answer_text

    async def add_task(
        self,
        ctx: RunContext[RuntimeState],
        sub_task_objective: str,
        capability: str,
        dependencies: list[int] | None = None,
        metadata: dict | None = None,
        parameters: dict | None = None,
    ) -> int:
        """Tool: Add Task (delegates to :class:`Scheduler`)."""
        async with self._plan_lock:
            return await self._supervisor_tools.add_task(
                ctx,
                sub_task_objective=sub_task_objective,
                capability=capability,
                dependencies=dependencies,
                metadata=metadata,
                parameters=parameters,
            )

    async def cancel_task(
        self,
        ctx: RunContext[RuntimeState],
        task_id: int,
        reason: str,
    ) -> str:
        """Tool: Cancel Task (delegates to :class:`Scheduler`)."""
        async with self._plan_lock:
            return await self._supervisor_tools.cancel_task(
                ctx, task_id=task_id, reason=reason
            )

    async def patch_task(
        self,
        ctx: RunContext[RuntimeState],
        task_id: int,
        sub_task_objective: Optional[str] = None,
        capability: Optional[str] = None,
        dependencies: Optional[List[int]] = None,
        parameters: dict | None = None,
    ) -> str:
        """Tool: Patch Task (delegates to :class:`Scheduler`)."""
        async with self._plan_lock:
            return await self._supervisor_tools.patch_task(
                ctx,
                task_id=task_id,
                sub_task_objective=sub_task_objective,
                capability=capability,
                dependencies=dependencies,
                parameters=parameters,
            )

    async def mark_final_task(
        self,
        ctx: RunContext[RuntimeState],
        task_id: int,
        reason: str | None = None,
    ) -> str:
        """Tool: Mark Final Task (delegates to :class:`Scheduler`)."""
        async with self._plan_lock:
            return await self._supervisor_tools.mark_final_task(
                ctx, task_id=task_id, reason=reason
            )

    def _is_terminal_status(self, status: TaskStatus) -> bool:
        """Delegate to :class:`Scheduler.is_terminal_status`."""
        return self._scheduler.is_terminal_status(status)

    async def _scheduler_pass(self, ctx: RuntimeState) -> str:
        """Delegate to :class:`Scheduler.scheduler_pass`."""
        return await self._scheduler.scheduler_pass(ctx)

    def _select_final_result(self, runtime_state: RuntimeState) -> TaskResult | None:
        """Delegate to :class:`Scheduler.select_final_result`."""
        return self._scheduler.select_final_result(runtime_state)

    def _build_deadlock_report(
        self, ctx: RuntimeState, decision: SupervisorDecision | None = None
    ) -> str:
        """Delegate to :class:`Scheduler.build_deadlock_report`."""
        return self._scheduler.build_deadlock_report(ctx, decision)

    @traced()
    async def run(self) -> PydanTaskRunResult:
        """Run the full DeepAgent control loop until completion or max steps.

        If a ``seed_plan`` was supplied at construction time, it is loaded into the
        runtime state before the supervisor loop begins.

        This method repeatedly:

        * Invokes the supervisor to decide which tasks to execute next.
        * Executes ready tasks in parallel via their associated sub-agents.
        * Sends results to the critic for QA and status updates.
        * Optionally checkpoints state between iterations.

        Returns:
            A ``DeepAgentRunResult`` containing the final output, the full plan,
            and the final ``RuntimeState``.
        """
        runtime_state = self._initialize_runtime_state(
            objective=self.objective, registry=self._capability_registry
        )

        self._apply_seed_plan(runtime_state)

        if self.resume:
            await self._replay_checkpoint(runtime_state)

        errors: list[str] = []
        no_progress_cycles = 0

        step_count = 0
        stop_execution = False
        while step_count < self._max_steps and not stop_execution:

            logger.info(f"--- Step {step_count} ---")

            if step_count == 0:
                logger.info("======= Planning Phase =======\n")

            # Best-effort global token budget enforcement.
            # Use getattr() so unit tests can construct DeepAgent without __init__.
            token_budget = getattr(self, "token_budget", None)
            if token_budget is not None and runtime_state.tokens_used >= token_budget:
                msg = (
                    f"Global token budget exceeded: tokens_used={runtime_state.tokens_used} "
                    f">= token_budget={token_budget}. Stopping execution."
                )
                logger.warning(msg)
                errors.append(msg)
                stop_execution = True
                break

            # Deterministic scheduler pass to normalize readiness and surface issues.
            self._last_scheduler_report = await self._scheduler_pass(runtime_state)

            current_instruction = (
                BOOTSTRAP_INSTRUCT
                if len(runtime_state.plan) == 0
                else ORCHESTRATION_INSTRUCT
            )
            supervisor_limits = self._make_usage_limits(
                total_tokens_limit=self._remaining_token_budget(runtime_state)
            )
            supervisor_run = await self._supervisor_agent.run(
                self._format_supervisor_input_prompt(runtime_state),
                deps=runtime_state,
                instructions=current_instruction,
                usage_limits=supervisor_limits,
            )
            self._accumulate_usage(runtime_state, supervisor_run, label="supervisor")
            supervisor_response = supervisor_run.output

            await self._record_event(
                "supervisor_decision",
                supervisor_response.model_dump(mode="json"),
            )

            if supervisor_response.all_tasks_completed:
                # Deterministic guardrail: do not allow "completion" unless the
                # task marked as final is actually COMPLETED.
                final_tasks = [
                    t
                    for t in runtime_state.plan.values()
                    if getattr(t, "is_final", False)
                ]

                completion_ok = True
                reasons: list[str] = []

                if not final_tasks:
                    completion_ok = False
                    reasons.append("no task is marked Final: True")
                elif len(final_tasks) > 1:
                    completion_ok = False
                    reasons.append(
                        f"multiple tasks are marked Final: True ({[t.task_id for t in final_tasks]})"
                    )
                else:
                    ft = final_tasks[0]
                    if ft.status != TaskStatus.COMPLETED:
                        completion_ok = False
                        reasons.append(
                            f"final task {ft.task_id} status is {ft.status.value!r} (expected 'completed')"
                        )
                    if ft.result is None:
                        completion_ok = False
                        reasons.append(f"final task {ft.task_id} has no TaskResult")

                if not completion_ok:
                    msg = (
                        "Supervisor returned all_tasks_completed=True, but completion invariants "
                        f"are not met: {', '.join(reasons)}. Overriding to continue."
                    )
                    logger.warning(msg)
                    self._last_scheduler_report = (
                        self._last_scheduler_report
                        + "\n\nCOMPLETION OVERRIDE (deterministic):\n- "
                        + msg
                    ).strip()

                    # Treat this as a no-progress cycle so we eventually fail-safe.
                    no_progress_cycles += 1
                    if self.checkpoint:
                        await self._checkpoint_state(runtime_state)
                    runtime_state.runtime_steps += 1
                    step_count += 1
                    continue

                logger.info(
                    "--- Supervisor declared completion and final task is completed. Ending execution loop. ---"
                )
                stop_execution = True
                break

            logger.info("--- Executing Tasks ---")
            # execute tasks that are ready to run and await responses
            task_results = await self._execute_ready_tasks(
                supervisor_response, runtime_state
            )

            # NOTE: `execute(...)` mutates the canonical TaskItem stored in `runtime_state.plan`
            # in-place (it receives the same object reference). Do NOT overwrite
            # `runtime_state.plan[task_id]` with returned TaskItems here; that can clobber
            # concurrent metadata updates (e.g. scratch notes/checkpoints).

            if len(task_results) == 0 and step_count != 0:
                # No tasks ran this cycle. This is not necessarily terminal in a
                # dynamic planner: we may be blocked on deps, have errored tasks
                # that need patching, or need the supervisor to add new nodes.
                no_progress_cycles += 1
                deadlock = self._build_deadlock_report(
                    runtime_state, supervisor_response
                )
                self._last_scheduler_report = (
                    self._last_scheduler_report + "\n\n" + deadlock
                ).strip()

                logger.info(
                    f"No executable tasks this cycle (no_progress_cycles={no_progress_cycles}). Continuing."
                )

                # Prevent infinite loops if the supervisor cannot make progress.
                if no_progress_cycles >= 3:
                    msg = (
                        "No progress after 3 consecutive cycles (no tasks executed). "
                        "Stopping to avoid infinite loop. "
                        "See SYSTEM SCHEDULER NOTES in the final cycle for details."
                    )
                    logger.warning(msg)
                    errors.append(msg)
                    stop_execution = True

                if self.checkpoint:
                    await self._checkpoint_state(runtime_state)

                runtime_state.runtime_steps += 1
                step_count += 1
                continue

            no_progress_cycles = 0

            logger.info(f"Number of tasks executed: {len(task_results)}")
            # go through responses and evaluate if they have completed the task
            for task_result in task_results or []:
                logger.info(f"--- Evaluating Task Result for {task_result.task_id} ---")

                critic_limits = self._make_usage_limits(
                    total_tokens_limit=self._remaining_token_budget(runtime_state)
                )
                qa_run = await self._critic_agent.run(
                    self._format_critic_input_prompt(task_result, runtime_state),
                    deps=runtime_state,
                    usage_limits=critic_limits,
                )
                self._accumulate_usage(runtime_state, qa_run, label="critic")
                qa_response = qa_run.output
                if self.verbose:
                    logger.info("--- QA Response ---")
                    logger.info(qa_response.model_dump_json(indent=2))

                task = runtime_state.plan[task_result.task_id]

                # deterministic transition based on critic
                await self.handle_critic_result(task, qa_response)

            if self.checkpoint:
                await self._checkpoint_state(runtime_state)

            runtime_state.runtime_steps += 1
            step_count += 1

        return_result = PydanTaskRunResult(
            objective=self.objective,
            final_result=self._select_final_result(runtime_state),
            plan=runtime_state.plan,
            runtime_state=runtime_state,
            errors=errors,
        )

        return return_result

    def _apply_seed_plan(self, runtime_state: RuntimeState) -> None:
        """Delegate to :class:`Scheduler.apply_seed_plan`."""
        self._scheduler.apply_seed_plan(runtime_state, self.seed_plan)

    def _dependencies_satisfied(self, step: TaskItem, ctx: RuntimeState) -> bool:
        """Delegate to :class:`Scheduler.dependencies_satisfied`."""
        return self._scheduler.dependencies_satisfied(step, ctx)

    async def _cascade_cancellations(self, ctx: RuntimeState) -> None:
        """Delegate to :class:`Scheduler.cascade_cancellations`."""
        await self._scheduler.cascade_cancellations(ctx)

    @traced(capture_input=False)
    async def _execute_ready_tasks(
        self, tasks: SupervisorDecision, ctx: RuntimeState
    ) -> list[TaskItem]:
        """Delegate to :class:`TaskExecutor.execute_ready_tasks`."""
        return await self._executor.execute_ready_tasks(tasks, ctx, self._plan_lock)

    @traced(run_type="task", capture_input=False)
    async def execute(
        self, capability: CapabilityRunner, step: TaskItem, runtime_state: RuntimeState
    ) -> TaskItem:
        """Delegate to :class:`TaskExecutor.execute`."""
        return await self._executor.execute(capability, step, runtime_state)

    async def update_task_status(
        self,
        ctx: RunContext[RuntimeState],
        task_id: int,
        status: TaskStatus,
    ) -> str:
        """Tool: Update Task Status (delegates to :class:`Scheduler`)."""
        async with self._plan_lock:
            return await self._supervisor_tools.update_task_status(
                ctx, task_id=task_id, status=status
            )

    async def handle_critic_result(self, task: TaskItem, review: TaskQAResult):
        """Delegate to :class:`CriticHandler.handle_critic_result`."""
        return await self._critic_handler.handle_critic_result(task, review)

    async def view_qa_report(
        self, ctx: RunContext[RuntimeState], task_id: int
    ) -> str:
        """Tool: View QA Report (delegates to :class:`Scheduler`)."""
        async with self._plan_lock:
            return await self._supervisor_tools.view_qa_report(
                ctx, task_id=task_id
            )
