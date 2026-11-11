"""Planning and sequential stopping.

required_runs(effect, alpha, power)
    Simulation-based task-count planning for the McNemar-on-consistency
    gate.  No closed form for pass^k planning exists in the source papers
    (2A research section 3.3), so we simulate under explicit, documented
    assumptions and report them alongside the answer.

sprt_stop(...)
    Wald sequential probability ratio test over the per-task consistency
    indicators of the candidate run.  Saves eval cost by stopping early.

consistency_settled(...)
    A task's consistency indicator z_x is settled as soon as it has one
    failure (it can never become 'consistently solved') or once all
    planned trials succeeded -- the basis for stopping per-task sampling.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Literal

from .compare import _p_value, select_test
from .model import EvalRun, group_by_task, task_counts


# ---------------------------------------------------------------------------
# planning


@dataclass
class PlanResult:
    n_tasks: int
    achieved_power: float
    alpha: float
    effect: float
    assumptions: list[str] = field(default_factory=list)
    power_curve: list[dict] = field(default_factory=list)


def _simulate_power(
    n_tasks: int,
    p_base: float,
    effect: float,
    alpha: float,
    rng: random.Random,
    n_sims: int = 400,
) -> float:
    p_new = min(1.0, p_base + effect)
    hits = 0
    for _ in range(n_sims):
        b = sum(1 for _ in range(n_tasks)
                if rng.random() >= p_base and rng.random() < p_new)
        c = sum(1 for _ in range(n_tasks)
                if rng.random() < p_base and rng.random() >= p_new)
        if b + c == 0:
            continue
        test_used = select_test(b + c, "auto")
        if _p_value(b, c, test_used) < alpha:
            hits += 1
    return hits / n_sims


def required_runs(
    effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
    p_base: float = 0.5,
    seed: int = 0,
    max_tasks: int = 5000,
) -> PlanResult:
    """Smallest number of paired tasks that detects ``effect`` at target power.

    Assumptions (documented, not hidden):
      * per-task consistency indicator z_x ~ Bernoulli(p) under each version;
      * the candidate's reliability is p_base + effect;
      * base and new indicators are independent across versions
        (pessimistic: real paired runs share task difficulty, which raises
        discordance contrast and usually makes real power >= simulated);
      * the gate test is the same auto exact/mid-p McNemar used by
        :func:`reliagate.compare.compare`.
    """
    if not 0 < effect < 1:
        raise ValueError("effect must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if not 0 <= p_base < 1 or p_base + effect > 1:
        raise ValueError("need 0 <= p_base and p_base + effect <= 1")

    assumptions = [
        f"z_x ~ Bernoulli({p_base}) under base, Bernoulli({p_base + effect}) under new",
        "base/new indicators simulated independent (conservative)",
        f"gate test: auto exact/mid-p McNemar at alpha={alpha}",
    ]

    # exponential search to bracket, then binary search inside the bracket
    rng_seed = seed
    curve = []

    def power_at(n_tasks: int, s: int) -> float:
        return _simulate_power(n_tasks, p_base, effect, alpha, random.Random(s))

    # find first n where power >= target using doubling
    found = None
    n = 1
    while n <= max_tasks:
        p = power_at(n, rng_seed + n)
        curve.append({"n_tasks": n, "power": p})
        if p >= power:
            found = n
            break
        n *= 2
    if found is None:
        return PlanResult(
            n_tasks=max_tasks,
            achieved_power=curve[-1]["power"] if curve else 0.0,
            alpha=alpha,
            effect=effect,
            assumptions=assumptions + [
                f"power {power} not reached within max_tasks={max_tasks}"
            ],
            power_curve=curve,
        )

    hi = found
    lo = max(1, found // 2)
    while lo < hi:
        mid = (lo + hi) // 2
        p = power_at(mid, rng_seed + mid)
        curve.append({"n_tasks": mid, "power": p})
        if p >= power:
            hi = mid
        else:
            lo = mid + 1

    curve.sort(key=lambda d: d["n_tasks"])
    final_power = power_at(lo, rng_seed + lo)
    best = PlanResult(
        n_tasks=lo,
        achieved_power=final_power,
        alpha=alpha,
        effect=effect,
        assumptions=assumptions,
        power_curve=curve,
    )
    return best


def plan(
    pilot: EvalRun,
    effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
    seed: int = 0,
    max_tasks: int = 5000,
) -> PlanResult:
    """Plan using a pilot run: p_base is estimated from the pilot's Pass^n."""
    counts = task_counts(pilot)
    consistent = sum(1 for c, n, _a in counts.values() if c == n)
    p_base = consistent / len(counts)
    result = required_runs(
        effect, alpha=alpha, power=power, p_base=p_base, seed=seed, max_tasks=max_tasks
    )
    result.assumptions.insert(0, f"p_base={p_base} estimated from pilot Pass^n")
    return result


# ---------------------------------------------------------------------------
# sequential stopping


def consistency_settled(failures_so_far: int, trials_done: int, n_planned: int) -> bool:
    """True when a task's z_x = 1[c_x == n] can no longer change.

    One failure settles z_x = 0 forever; all-success through n_planned
    settles z_x = 1.  Stop sampling such tasks in sequential mode.
    """
    if n_planned <= 0:
        raise ValueError("n_planned must be > 0")
    if failures_so_far > 0:
        return True
    return trials_done >= n_planned


@dataclass
class SprtResult:
    decision: Literal["accept", "reject", "continue"]
    llr: float
    upper: float
    lower: float
    n_tasks: int
    notes: list[str] = field(default_factory=list)


def sprt_stop(
    base: EvalRun,
    cand_so_far: EvalRun,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.10,
    max_tasks: int = 200,
) -> SprtResult:
    """Wald SPRT over the candidate's per-task consistency indicators.

    H0: new reliability p0 = base's Pass^n (no change).
    H1: new reliability p1.
    Each task contributes log[ P1(z_x) / P0(z_x) ]; stop when the LLR
    crosses ln((1-beta)/alpha) (accept H1: candidate is better) or
    ln(beta/(1-alpha)) (reject H1).  ``continue`` at the max_tasks cap is
    INCONCLUSIVE -- callers must treat it as such, never as a pass.
    """
    base_counts = task_counts(base)
    p0 = sum(1 for c, n, _a in base_counts.values() if c == n) / len(base_counts)
    if not 0 < p0 < 1 or not 0 < p1 < 1 or p1 == p0:
        raise ValueError(
            f"need 0 < p0 < 1, 0 < p1 < 1, p1 != p0 (got p0={p0}, p1={p1})"
        )

    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    llr = 0.0
    n_tasks = 0
    for _tid, trials in group_by_task(cand_so_far).items():
        c = sum(1 for t in trials if t.success is True)
        n = len(trials)
        z = 1 if c == n and n > 0 else 0
        llr += z * math.log(p1 / p0) + (1 - z) * math.log((1 - p1) / (1 - p0))
        n_tasks += 1
        if llr >= upper:
            return SprtResult("accept", llr, upper, lower, n_tasks)
        if llr <= lower:
            return SprtResult("reject", llr, upper, lower, n_tasks)

    notes = []
    if n_tasks >= max_tasks:
        notes.append(
            f"max_tasks={max_tasks} reached without crossing a boundary: "
            "INCONCLUSIVE, do not treat as pass"
        )
    return SprtResult("continue", llr, upper, lower, n_tasks, notes)
