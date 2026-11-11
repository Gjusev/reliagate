"""Reliability metrics -- pure Python, zero dependencies.

All combinatorics use :func:`math.comb` on exact integers, so estimators
match the papers' closed forms without floating-point accumulation error.

References:
    [tau-bench]   Yao et al., arXiv:2406.12045, section 3 (pass^k, pass@k).
    [codex]       Chen et al., arXiv:2107.03374, section 2.1 Eq. 1 +
                  Appendix A (unbiased pass@k; bias of 1-(1-p_hat)^k).
    [reliability] arXiv:2604.17849, section 2.2 (consistency classes,
                  z_x = 1[c_x == n]).
    [precipice]   arXiv:2605.08261, section 4.2 Eq. 1 (Wilson interval),
                  section 4.3 (bootstrap over tasks, never over rollouts).
    [rule3]       Hanley & Lippman-Hand 1983 (rule of three).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .model import EvalRun, consistency_class, task_counts


@dataclass
class Estimate:
    """A point estimate with its interval and provenance."""

    value: float
    ci: Optional[tuple[float, float]] = None
    method: str = "point"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.ci is not None:
            lo, hi = self.ci
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(f"CI out of [0,1]: {self.ci}")


# ---------------------------------------------------------------------------
# per-task point estimators (exact integer combinatorics)


def task_pass_pow_k(c: int, n: int, k: int) -> float:
    """Per-task pass^k = C(c, k) / C(n, k).  [tau-bench] section 3."""
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    return math.comb(c, k) / math.comb(n, k)


def task_pass_at_k(c: int, n: int, k: int) -> float:
    """Per-task unbiased pass@k = 1 - C(n-c, k) / C(n, k).  [codex] Eq. 1."""
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# run-level aggregates


def _check_k(run: EvalRun, k: int, drop_abstained: bool) -> None:
    for tid, (c, n, _a) in task_counts(run, drop_abstained).items():
        if n < k:
            raise ValueError(
                f"task {tid!r} has only {n} denominator trials, fewer than "
                f"k={k}. Provide more attempts for every task "
                f"(drop_abstained={drop_abstained})."
            )


def pass_pow_k(
    run: EvalRun, k: int, drop_abstained: bool = False
) -> float:
    """Run-level pass^k: mean over tasks of C(c_x, k)/C(n_x, k). [tau-bench].

    pass^1 == pass@1 == mean c_x/n_x (the tau-bench identity, section 3).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    _check_k(run, k, drop_abstained)
    counts = task_counts(run, drop_abstained)
    return sum(task_pass_pow_k(c, n, k) for c, n, _a in counts.values()) / len(counts)


def pass_at_k(run: EvalRun, k: int, drop_abstained: bool = False) -> float:
    """Run-level unbiased pass@k.  [codex] section 2.1 Eq. 1.

    NOTE: the naive ``1 - (1 - p_hat)**k`` estimator is a consistent
    *underestimate* -- never use it.  [codex] Appendix A.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    _check_k(run, k, drop_abstained)
    counts = task_counts(run, drop_abstained)
    return sum(task_pass_at_k(c, n, k) for c, n, _a in counts.values()) / len(counts)


def consistency(run: EvalRun) -> dict:
    """Counts of consistently / sometimes / never solved tasks. [reliability]."""
    classes = {}
    for tid, (c, n, _a) in task_counts(run).items():
        classes[tid] = consistency_class(c, n)
    counts = {"consistent": 0, "sometimes": 0, "never": 0, "no_trials": 0}
    for cls in classes.values():
        counts[cls] += 1
    total_tasks = sum(counts.values())
    return {
        "counts": counts,
        "total_tasks": total_tasks,
        "pass_pow_n": counts["consistent"] / total_tasks if total_tasks else 0.0,
        "per_task": classes,
    }


def per_task(run: EvalRun, k: int = 1, drop_abstained: bool = False) -> list[dict]:
    """One row per task: counts, pass^k point estimate, rule-of-three bound.

    The rule-of-three upper bound on the per-task *failure* probability
    (3/n at 95%) applies when c == n; it exists so nobody mistakes a
    3/3 = 1.0 pass^k for certainty.  [rule3]
    """
    _check_k(run, k, drop_abstained)
    rows = []
    for tid, (c, n, abstained) in task_counts(run, drop_abstained).items():
        row = {
            "task_id": tid,
            "c": c,
            "n": n,
            "abstained": abstained,
            "pass_pow_k": task_pass_pow_k(c, n, k),
            "class": consistency_class(c, n),
        }
        if c == n:
            row["failure_upper_bound_95"] = 3.0 / n
        else:
            row["failure_upper_bound_95"] = 1.0 - task_pass_pow_k(c, n, 1)
        rows.append(row)
    rows.sort(key=lambda r: r["task_id"])
    return rows


# ---------------------------------------------------------------------------
# confidence intervals


def wilson_ci(
    successes: int, total: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. [precipice] Eq. 1.

    Chosen over the Wald interval because Wald collapses to a zero-width
    interval at p_hat = 0 or 1 -- exactly where reliability metrics live.
    """
    if total <= 0:
        raise ValueError("total must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 <= successes <= total:
        raise ValueError("successes out of range")
    z = _z_quantile(1.0 - alpha / 2.0)
    p_hat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt(
        p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total)
    )
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    if lo < 1e-12:
        lo = 0.0  # snap float dust; true bound is exactly 0 here
    if hi > 1.0 - 1e-12:
        hi = 1.0
    return (lo, hi)


def _z_quantile(prob: float) -> float:
    """Inverse standard-normal CDF via bisection on math.erf (no scipy)."""
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < prob:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def wilson_indicator(
    run: EvalRun, alpha: float = 0.05
) -> Estimate:
    """Wilson CI for Pass^n (fraction of consistently solved tasks).

    The task is the unit of inference, so a plain binomial Wilson interval
    applies.  [reliability] z_x; [precipice] section 4.2 with R = |tasks|.
    """
    cons = consistency(run)
    total = cons["total_tasks"]
    succ = cons["counts"]["consistent"]
    lo, hi = wilson_ci(succ, total, alpha)
    return Estimate(
        value=cons["pass_pow_n"],
        ci=(lo, hi),
        method="wilson on per-task indicator z_x = 1[c_x == n]",
        notes=[f"{total} tasks, {succ} consistently solved"],
    )


# ---------------------------------------------------------------------------
# bootstrap over tasks (never over trials within a task)


def bootstrap_pass_pow_k(
    run: EvalRun,
    k: int,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    drop_abstained: bool = False,
) -> Estimate:
    """Percentile CI for pass^k by resampling *tasks* with replacement.

    Runs within a task are correlated and must NOT be resampled
    independently: a rollout-only bootstrap reached only 17% coverage under
    heterogeneous conditions, while task-level resampling restored nominal
    coverage.  [precipice] sections 2.2, 4.3, 6.
    """
    _check_k(run, k, drop_abstained)
    values = [
        task_pass_pow_k(c, n, k)
        for c, n, _a in task_counts(run, drop_abstained).values()
    ]
    point = sum(values) / len(values)
    rng = random.Random(seed)
    m = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()

    def _quantile(q: float) -> float:
        nb = len(means)
        idx = min(nb - 1, max(0, int(round(q * (nb - 1)))))
        return means[idx]

    lo = _quantile(alpha / 2.0)
    hi = _quantile(1.0 - alpha / 2.0)
    return Estimate(
        value=point,
        ci=(lo, hi),
        method=f"percentile bootstrap over tasks (B={n_boot}, seed={seed})",
        notes=[
            "tasks resampled with replacement; trials within a task never resampled",
        ],
    )


def pass_k(
    run: EvalRun,
    k: int,
    ci: str | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    n_boot: int = 1000,
    drop_abstained: bool = False,
) -> Estimate:
    """pass^k with an optional interval; the front-door metric API.

    ci=None            -> point estimate only.
    ci="bootstrap_tasks" -> percentile bootstrap over tasks.
    ci="wilson_indicator" -> Wilson CI on the Pass^n task indicator instead
        (defensible because Pass^n IS a binomial proportion over tasks).
    """
    if ci is None:
        return Estimate(value=pass_pow_k(run, k, drop_abstained), method="point")
    if ci == "bootstrap_tasks":
        return bootstrap_pass_pow_k(
            run, k, n_boot=n_boot, seed=seed, alpha=alpha, drop_abstained=drop_abstained
        )
    if ci == "wilson_indicator":
        return wilson_indicator(run, alpha)
    raise ValueError(f"unknown ci mode: {ci!r}")
