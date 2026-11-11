"""reliagate data model.

One :class:`Run` is a single execution of one task.  ``success=None`` means
the trial abstained or errored for infrastructure reasons; it is NEVER
silently counted as a failure.  Every summary produced by this library
states explicitly how abstentions were treated.

Convention (chosen deliberately, and documented in README):
    * abstained trials are counted in the denominator ``n``;
    * they are never counted in the successes ``c``;
    * so they depress the metric -- the conservative direction for a gate;
    * every report carries the abstention counts so nothing hides.

This avoids the ambiguity tau2-bench stumbled into (issue #497), where
infra-error handling silently changed task weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Run:
    """One execution of one task.

    success: True / False, or None for an abstained or infra-errored trial.
    attempt: 0-based index of the attempt within the task.
    """

    task_id: str
    attempt: int
    success: Optional[bool]
    version: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool):
            raise ValueError("attempt must be an int")
        if self.success is not None and not isinstance(self.success, bool):
            raise ValueError("success must be True, False or None")
        if self.attempt < 0:
            raise ValueError("attempt must be >= 0")


@dataclass
class EvalRun:
    """All trials of one evaluated version."""

    name: str
    runs: list[Run] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[tuple[str, int]] = set()
        for run in self.runs:
            key = (run.task_id, run.attempt)
            if key in seen:
                raise ValueError(
                    f"duplicate trial for task {run.task_id!r} attempt {run.attempt}; "
                    "each (task_id, attempt) pair must appear exactly once"
                )
            seen.add(key)


def group_by_task(run: EvalRun) -> dict[str, list[Run]]:
    """Return {task_id: [runs sorted by attempt]}."""
    grouped: dict[str, list[Run]] = {}
    for r in run.runs:
        grouped.setdefault(r.task_id, []).append(r)
    for trials in grouped.values():
        trials.sort(key=lambda r: r.attempt)
    return grouped


def task_counts(run: EvalRun, drop_abstained: bool = False) -> dict[str, tuple[int, int, int]]:
    """Per-task ``(c, n, abstained)`` counts.

    ``c``  = successful trials.
    ``n``  = trials counted in the denominator: all trials by default;
             only decisive trials when ``drop_abstained=True``.
    ``abstained`` = trials with ``success is None`` (always reported,
             even when dropped from ``n``).
    """
    out: dict[str, tuple[int, int, int]] = {}
    for task_id, trials in group_by_task(run).items():
        c = sum(1 for t in trials if t.success is True)
        abstained = sum(1 for t in trials if t.success is None)
        n = len(trials) if not drop_abstained else len(trials) - abstained
        out[task_id] = (c, n, abstained)
    return out


def abstention_report(run: EvalRun) -> dict:
    """How abstentions were handled -- attached to every report."""
    total = len(run.runs)
    abstained = sum(1 for r in run.runs if r.success is None)
    tasks = {tid: a for tid, (c, n, a) in task_counts(run).items() if a > 0}
    return {
        "trials_total": total,
        "trials_abstained": abstained,
        "tasks_with_abstentions": len(tasks),
        "per_task_abstained": tasks,
        "treatment": "counted in denominator n, excluded from successes c "
        "(conservative); never silently treated as failure",
    }


def consistency_class(c: int, n: int) -> str:
    """Three-way per-task class (arXiv:2604.17849 section 2.2).

    'consistent' requires c == n with no abstentions in n: every single
    trial of the task succeeded.  Abstentions therefore keep a task out of
    'consistent' -- a task proven reliable only on decisive evidence.
    """
    if n <= 0:
        return "no_trials"
    if c == n:
        return "consistent"
    if c == 0:
        return "never"
    return "sometimes"


def consistency_classes(run: EvalRun) -> dict[str, str]:
    """Per-task consistency class: 'consistent' | 'sometimes' | 'never'."""
    return {
        tid: consistency_class(c, n) for tid, (c, n, _a) in task_counts(run).items()
    }
