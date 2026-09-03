import yaml
from pathlib import Path
from typing import Any, Dict, List, Set

from pydantic import ValidationError

from pydantask.models import Plan, WorkflowYamlConfig


def get_incremented_path(
    base_name: str, extension: str, directory: Path = Path("checkpoints")
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    counter = 1
    # Initial path attempt: checkpoints/state_1.json
    target_path = directory / f"{base_name}_{counter}.{extension}"

    # Keep incrementing until we find a filename that doesn't exist
    while target_path.exists():
        counter += 1
        target_path = directory / f"{base_name}_{counter}.{extension}"

    return target_path


def _ensure_dag_is_valid(plan: Plan) -> None:
    """Validate DAG properties that Pydantic types alone don't enforce."""
    tasks = list(plan.tasks or [])

    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate task_id values in workflow: {dupes}")

    task_ids: Set[int] = set(ids)
    for t in tasks:
        for dep in t.sub_task_dependencies or []:
            if dep not in task_ids:
                raise ValueError(
                    f"Task {t.task_id} depends on missing task_id {dep}. "
                    "All dependencies must reference an existing task_id in the workflow."
                )
            if dep == t.task_id:
                raise ValueError(f"Task {t.task_id} cannot depend on itself")

    # Cycle detection via DFS.
    graph: Dict[int, List[int]] = {
        t.task_id: list(t.sub_task_dependencies or []) for t in tasks
    }

    visiting: Set[int] = set()
    visited: Set[int] = set()

    def dfs(node: int, stack: List[int]) -> None:
        if node in visited:
            return
        if node in visiting:
            # found a cycle; report a helpful path
            cycle_start = stack.index(node) if node in stack else 0
            cycle_path = stack[cycle_start:] + [node]
            raise ValueError(
                f"Dependency cycle detected: {' -> '.join(map(str, cycle_path))}"
            )

        visiting.add(node)
        stack.append(node)
        for dep in graph.get(node, []):
            dfs(dep, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for tid in graph.keys():
        dfs(tid, [])


async def import_yaml_workflow(
    path: str | Path, *, auto_mark_final: bool = True
) -> Plan:
    """Load a pre-defined workflow (task DAG) from a YAML file and validate it.

    Returns:
        Plan: A validated `Plan` suitable to pass as `seed_plan=...` to `DeepAgent`.

    Notes:
        - This is intended for user-provided *seed plans* (pre-defined DAGs).
        - Tasks default to `status=pending` so the deterministic scheduler can
          promote them to READY when dependencies are satisfied.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Validate against the strict user-facing YAML schema and convert into the
    # canonical Plan/TaskItem models.
    try:
        cfg = WorkflowYamlConfig.model_validate(raw)
        plan = cfg.to_plan()
    except ValidationError as e:
        raise ValueError(
            f"Invalid workflow YAML at {str(path)!r}.\n\nPydantic validation error:\n{str(e)}"
        ) from e

    _ensure_dag_is_valid(plan)

    # Enforce/assist with the 'final task' invariant expected by DeepAgent's completion guardrail.
    final_tasks = [t for t in plan.tasks if getattr(t, "is_final", False)]
    if len(final_tasks) > 1:
        raise ValueError(
            f"Workflow YAML marks multiple tasks as final: {[t.task_id for t in final_tasks]}. "
            "Mark exactly one task with `is_final: true`."
        )

    if len(final_tasks) == 0 and auto_mark_final:
        # Deterministic fallback: mark the max task_id as final.
        last = max(plan.tasks, key=lambda t: t.task_id)
        last.is_final = True

    return plan
