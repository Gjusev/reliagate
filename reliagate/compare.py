"""Paired comparison of two versions: McNemar on the consistency indicator.

Design (arXiv:2604.17849 section 2.2): for each task x compute the binary
indicator z_x = 1[c_x == n] ("consistently solved") under base and under
the candidate, then test the paired binary outcomes:

    b = number of tasks that GAINED consistent solvability (base 0 -> new 1)
    c = number of tasks that LOST it                (base 1 -> new 0)

Test selection ("auto"): the asymptotic chi-squared statistic
(b-c)^2/(b+c) is anti-conservative when discordant pairs are few, which is
the normal regime for agent evals.  Following standard practice
(Fagerland, Lydersen & Laake 2013, Stat Med 32:3086):

    exact two-sided binomial McNemar   when b + c <  25
    mid-p variant                      when b + c >= 25

The mid-p variant assigns the observed table half its exact probability,
correcting the conservativeness of the discrete exact test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import EvalRun, consistency_classes, group_by_task, task_counts


# ---------------------------------------------------------------------------
# tests


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant pairs."""
    if b < 0 or c < 0:
        raise ValueError("b and c must be >= 0")
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2**m
    return min(1.0, 2.0 * tail)


def mcnemar_midp(b: int, c: int) -> float:
    """Mid-p McNemar: exact p minus half the probability of the observed tail end.

    p_mid = 2 * P(X < min(b,c)) + P(X = min(b,c)) = p_exact - P(X = min(b,c)).
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be >= 0")
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    p = mcnemar_exact(b, c) - math.comb(m, k) / 2**m
    return max(0.0, min(1.0, p))


def select_test(discordant: int, test: str = "auto") -> str:
    if test in ("exact", "mcnemar_exact"):
        return "mcnemar_exact"
    if test in ("midp", "mcnemar_midp"):
        return "mcnemar_midp"
    if test == "auto":
        return "mcnemar_exact" if discordant < 25 else "mcnemar_midp"
    raise ValueError(f"unknown test: {test!r}")


def _p_value(b: int, c: int, which: str) -> float:
    return mcnemar_exact(b, c) if which == "mcnemar_exact" else mcnemar_midp(b, c)


# ---------------------------------------------------------------------------
# comparison


@dataclass
class ComparisonResult:
    b: int                      # tasks gained consistent solvability
    c: int                      # tasks lost it
    b_minus_c: int              # signed effect (direction + magnitude)
    p_value: float
    test_used: str
    n_tasks: int
    discordant: int
    base_consistent: int
    new_consistent: int
    base_pass_pow_n: float
    new_pass_pow_n: float
    class_changes: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_significant_at_0_05(self) -> bool:
        return self.p_value < 0.05


def _validate_paired(base: EvalRun, new: EvalRun) -> None:
    base_tasks = set(group_by_task(base))
    new_tasks = set(group_by_task(new))
    if base_tasks != new_tasks:
        only_base = sorted(base_tasks - new_tasks)
        only_new = sorted(new_tasks - base_tasks)
        raise ValueError(
            "paired comparison requires identical task sets; "
            f"only in base: {only_base}; only in new: {only_new}"
        )


def compare(
    base: EvalRun,
    new: EvalRun,
    test: str = "auto",
) -> ComparisonResult:
    """Paired McNemar on per-task 'consistently solved' indicators."""
    _validate_paired(base, new)
    base_classes = consistency_classes(base)
    new_classes = consistency_classes(new)

    b = sum(
        1 for t in base_classes
        if base_classes[t] != "consistent" and new_classes[t] == "consistent"
    )
    c = sum(
        1 for t in base_classes
        if base_classes[t] == "consistent" and new_classes[t] != "consistent"
    )
    discordant = b + c
    test_used = select_test(discordant, test)

    changes = []
    for tid in sorted(base_classes):
        if base_classes[tid] != new_classes[tid]:
            changes.append(
                {"task_id": tid, "base": base_classes[tid], "new": new_classes[tid]}
            )

    base_counts = task_counts(base)
    new_counts = task_counts(new)
    base_consistent = sum(1 for v in base_classes.values() if v == "consistent")
    new_consistent = sum(1 for v in new_classes.values() if v == "consistent")
    n_tasks = len(base_classes)

    notes = []
    for name, counts in (("base", base_counts), ("new", new_counts)):
        abst = {tid: a for tid, (_c, _n, a) in counts.items() if a > 0}
        if abst:
            notes.append(
                f"{name} run has abstained trials counted in n: {abst}"
            )

    return ComparisonResult(
        b=b,
        c=c,
        b_minus_c=b - c,
        p_value=_p_value(b, c, test_used),
        test_used=test_used,
        n_tasks=n_tasks,
        discordant=discordant,
        base_consistent=base_consistent,
        new_consistent=new_consistent,
        base_pass_pow_n=base_consistent / n_tasks,
        new_pass_pow_n=new_consistent / n_tasks,
        class_changes=changes,
        notes=notes,
    )
