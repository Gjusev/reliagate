"""nitelos verdicts adapter (optional, lazy import).

Maps nitelos verdict objects to reliagate Run trials.  nitelos is NOT a
dependency of the core; import this module only when nitelos is installed.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..model import Run


def _verdict_to_success(verdict: Any) -> bool | None:
    """Map a nitelos verdict to True/False/None.

    Known mappings (documented, best-effort -- nitelos is not pinned here):
      * truthy string 'pass'/'passed'/'success'/'true' -> True
      * 'fail'/'failed'/'false' -> False
      * 'error'/'abstain'/'skipped'/None -> None (abstained, never a failure)
    """
    if verdict is None:
        return None
    if isinstance(verdict, bool):
        return verdict
    text = str(verdict).strip().lower()
    if text in ("pass", "passed", "success", "true", "1"):
        return True
    if text in ("fail", "failed", "false", "0"):
        return False
    return None  # error/abstain/unknown -> abstained


def verdicts_to_runs(
    verdicts: Iterable[Any],
    *,
    task_ids: Iterable[str] | None = None,
    version: str | None = None,
) -> list[Run]:
    """Convert an iterable of nitelos verdicts to reliagate Runs.

    ``task_ids`` must align with ``verdicts`` when the verdicts do not
    carry their own task identity (attribute ``task_id`` or mapping key).
    """
    task_list = list(task_ids) if task_ids is not None else None
    runs: list[Run] = []
    counters: dict[str, int] = {}
    for i, verdict in enumerate(verdicts):
        task_id: str | None = None
        raw = verdict
        if isinstance(verdict, dict):
            task_id = verdict.get("task_id")
            raw = verdict.get("verdict", verdict.get("passed", verdict.get("result")))
        elif hasattr(verdict, "task_id"):
            task_id = getattr(verdict, "task_id", None)
            for attr in ("verdict", "passed", "result"):
                if hasattr(verdict, attr):
                    raw = getattr(verdict, attr)
                    break
        if task_id is None:
            if task_list is None:
                raise ValueError(
                    "verdicts carry no task_id and no task_ids were provided"
                )
            task_id = task_list[i]
        attempt = counters.get(task_id, 0)
        counters[task_id] = attempt + 1
        runs.append(
            Run(
                task_id=task_id,
                attempt=attempt,
                success=_verdict_to_success(raw),
                version=version,
                metadata={"source": "nitelos"},
            )
        )
    return runs
