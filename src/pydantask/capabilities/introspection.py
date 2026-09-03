from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

# Names your runner/executor can inject automatically; these should *not* be
# treated as supervisor-supplied inputs.
INJECTED_PARAM_NAMES: set[str] = {
    # prompt aliases
    "prompt",
    "text",
    "input",
    "question",
    "query",
    # deps/state/task
    "deps",
    "task_deps",
    "runtime_state",
    "state",
    "runtime",
    "task",
    "step",
    # limits
    "usage_limits",
    "limits",
    # parameters dict passthrough
    "parameters",
    "params",
    "task_parameters",
}


def _type_to_str(tp: Any) -> str:
    """Best-effort type hint pretty-printer for prompts."""
    if tp is inspect._empty or tp is None:
        return "Any"

    # Resolve Optional[T] / Union[T, None]
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return f"Optional[{_type_to_str(args[0])}]"
        return "Union[" + ", ".join(_type_to_str(a) for a in args) + "]"

    if origin is not None:
        args = get_args(tp)
        origin_name = getattr(origin, "__name__", str(origin))
        if args:
            return f"{origin_name}[" + ", ".join(_type_to_str(a) for a in args) + "]"
        return origin_name

    # Builtins / classes
    name = getattr(tp, "__name__", None)
    if name:
        return name

    return str(tp)


def unwrap_callable(tool_func: Any) -> Optional[Callable[..., Any]]:
    """Try to recover the original callable from a capability implementation.

    Supports:
      - raw callables
      - runner wrappers that expose `.func` (AsyncFuncRunner/SyncFuncRunner patterns)

    Returns None for Agent-like runners.
    """
    if tool_func is None:
        return None

    # Runner wrappers frequently store the original function as `.func`.
    inner = getattr(tool_func, "func", None)
    if callable(inner):
        return inner

    # If the tool_func itself is callable (and not just `.run(...)`), use it.
    if callable(tool_func):
        return tool_func

    return None


def callable_input_schema(
    func: Callable[..., Any],
    *,
    injected_param_names: set[str] | None = None,
) -> dict[str, Any]:
    """Derive a small JSON-schema-like input contract from a Python callable.

    The schema is intended for *prompting the supervisor*, not strict validation.

    Parameters named in `injected_param_names` are excluded (they are supplied by
    the runtime, not the supervisor).

    Returns:
      {
        "name": "function_name",
        "required": ["arg1", ...],
        "optional": ["arg2", ...],
        "properties": {
          "arg1": {"type": "str", "default": null},
          "arg2": {"type": "int", "default": 3},
        },
        "notes": "..."
      }
    """
    injected = injected_param_names or INJECTED_PARAM_NAMES

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {
            "name": getattr(func, "__name__", "<callable>"),
            "required": [],
            "optional": [],
            "properties": {},
            "notes": "Signature not introspectable; treat as prompt+deps callable.",
        }

    # Resolve annotations robustly (handles postponed eval / forward refs).
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        hints = {}

    required: list[str] = []
    optional: list[str] = []
    props: dict[str, Any] = {}

    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # Can't enumerate *args/**kwargs meaningfully.
            continue

        if name in injected:
            continue

        ann = hints.get(name, p.annotation)
        type_str = _type_to_str(ann)

        if p.default is inspect._empty:
            required.append(name)
            props[name] = {"type": type_str, "default": None}
        else:
            optional.append(name)
            props[name] = {"type": type_str, "default": p.default}

    return {
        "name": getattr(func, "__name__", "<callable>"),
        "required": required,
        "optional": optional,
        "properties": props,
        "notes": (
            "Supervisor should provide these values in TaskItem.parameters={...}. "
            "(The runtime injects prompt/deps/task/runtime_state automatically when requested.)"
        ),
    }


def format_callable_inputs_for_prompt(
    schema: dict[str, Any], *, max_items: int = 8
) -> str:
    """Format a callable_input_schema(...) result into a compact prompt string."""
    required = list(schema.get("required") or [])
    optional = list(schema.get("optional") or [])
    props: dict[str, Any] = schema.get("properties") or {}

    def fmt(names: list[str]) -> str:
        chunks: list[str] = []
        for n in names[:max_items]:
            t = (props.get(n) or {}).get("type") or "Any"
            chunks.append(f"{n}:{t}")
        if len(names) > max_items:
            chunks.append(f"...(+{len(names) - max_items} more)")
        return ", ".join(chunks) if chunks else "<none>"

    return f"inputs(required): {fmt(required)}; optional: {fmt(optional)}"
