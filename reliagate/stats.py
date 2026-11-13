"""Deferred inference statistics (ROADMAP items 2-4 and 7).

Pure Python, zero dependencies.  Four tools that complement the McNemar
core in :mod:`reliagate.compare`:

* :func:`wilcoxon_signed_rank` -- Wilcoxon signed-rank test on paired
  per-task differences ``d_x = c_x_new - c_x_base`` (average ranks, ties
  handled; exact enumeration for small n, tie-corrected normal
  approximation otherwise).  arXiv:2604.17849 section 2.2 uses this as
  the secondary test next to McNemar because McNemar discards partial
  consistency shifts (e.g. 1 -> 2 of 3).
* :func:`holm_adjust` / :func:`benjamini_hochberg_adjust` -- classical
  step-down / step-up corrections over a family of p-values (one family
  = one task-category comparison, ROADMAP item 3).  Holm for
  confirmatory gates, BH for exploratory dashboards.
* :func:`hierarchical_bootstrap` -- percentile CI resampling scenarios
  then tasks within scenarios; the per-task aggregate (the "runs" level)
  is held FIXED during resampling, per arXiv:2605.08261 section 4.3.
  Resampling rollouts within a task independently reached only 17%
  interval coverage in that paper; the hierarchy must be respected.
* :func:`newcombe_paired` -- Newcombe hybrid-score interval for the
  difference of two PAIRED proportions, the recommended interval
  companion to McNemar (which alone yields no interval).

References:
    [reliability]  arXiv:2604.17849, section 2.2 (Wilcoxon on d_x).
    [multiplicity] Holm 1979; Benjamini & Hochberg 1995.
    [precipice]    arXiv:2605.08261, section 4.3 (hierarchical bootstrap).
    [newcombe]     Newcombe 1998, Stat Med 17:2635 (hybrid score method
                   10, paired case: Wilson intervals for the marginals
                   combined with the observed within-pair correlation).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .metrics import Estimate, wilson_ci


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank (paired differences)


@dataclass
class WilcoxonResult:
    n_pairs: int                 # input differences
    n_nonzero: int               # after dropping zeros (the test's n)
    w_plus: float                # sum of ranks with d > 0
    w_minus: float               # sum of ranks with d < 0
    w: float                     # min(w_plus, w_minus)
    p_value: float
    method: str                  # "exact" | "normal_approx" | "degenerate"
    z_statistic: Optional[float] = None
    notes: list[str] = field(default_factory=list)


def average_ranks(values: list[float]) -> list[float]:
    """Ranks (1-based) with ties assigned their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(
    diffs: list[float],
    alternative: str = "two-sided",
    exact: Optional[bool] = None,
) -> WilcoxonResult:
    """Wilcoxon signed-rank test on paired differences (zero-dependency).

    Zeros are dropped (standard signed-rank handling); ties share average
    ranks.  ``exact=True`` enumerates all 2^n sign assignments of the
    ranks (n <= 20 only); ``exact=None`` picks exact for n_nonzero <= 20
    and the tie-corrected normal approximation (with continuity
    correction) otherwise.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    notes: list[str] = []
    if n == 0:
        return WilcoxonResult(
            n_pairs=len(diffs), n_nonzero=0, w_plus=0.0, w_minus=0.0, w=0.0,
            p_value=1.0, method="degenerate",
            notes=["all differences are zero: the test is undefined; "
                   "reporting p=1.0 (no evidence of any shift)"],
        )

    absnz = [abs(d) for d in nz]
    ranks = average_ranks(absnz)
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nz) if d < 0)
    w_obs = min(w_plus, w_minus)

    use_exact = n <= 20 if exact is None else exact
    if use_exact:
        if n > 20:
            raise ValueError("exact signed-rank needs n <= 20 (2^n enumeration)")
        # Null distribution: every sign assignment of the ranks is equally
        # likely.  W = min(W+, W-) is small when signs are imbalanced, so
        # P(W <= w_obs) IS the two-sided p-value; one-sided variants use
        # W+ (or W-) directly.
        dist: dict[float, int] = {}
        dist_plus: dict[float, int] = {}
        for signs in range(1 << n):
            s_plus = 0.0
            s_minus = 0.0
            for i in range(n):
                if signs >> i & 1:
                    s_plus += ranks[i]
                else:
                    s_minus += ranks[i]
            wmin = min(s_plus, s_minus)
            dist[wmin] = dist.get(wmin, 0) + 1
            dist_plus[s_plus] = dist_plus.get(s_plus, 0) + 1
        total = 1 << n
        if alternative == "two-sided":
            p = sum(v for k, v in dist.items() if k <= w_obs) / total
        elif alternative == "greater":  # H1: differences tend positive
            p = sum(v for k, v in dist_plus.items() if k >= w_plus) / total
        else:                           # H1: differences tend negative
            p = sum(v for k, v in dist_plus.items() if k <= w_plus) / total
        method = "exact"
        z = None
    else:
        exp_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        # tie correction: one term per group of equal |d|
        counts: dict[float, int] = {}
        for a in absnz:
            counts[a] = counts.get(a, 0) + 1
        tie_term = sum(t**3 - t for t in counts.values()) / 48.0
        var_w -= tie_term
        sigma = math.sqrt(var_w)
        if sigma == 0.0:
            return WilcoxonResult(
                n_pairs=len(diffs), n_nonzero=n, w_plus=w_plus, w_minus=w_minus,
                w=w_obs, p_value=1.0, method="normal_approx",
                notes=["all |differences| identical (degenerate ranks)"],
            )
        if alternative == "two-sided":
            z = (abs(w_obs - exp_w) - 0.5) / sigma
            p = 2.0 * (1.0 - _normal_cdf(z))
        elif alternative == "greater":
            z = (w_plus - exp_w - 0.5) / sigma
            p = 1.0 - _normal_cdf(z)
        else:
            z = (w_plus - exp_w + 0.5) / sigma
            p = 1.0 - _normal_cdf(z)
        method = "normal_approx"
        notes.append(f"tie correction term {tie_term:.4f} subtracted from variance")
    notes.append(f"{n} non-zero differences ({len(diffs) - n} zeros dropped)")
    return WilcoxonResult(
        n_pairs=len(diffs), n_nonzero=n, w_plus=w_plus, w_minus=w_minus,
        w=w_obs, p_value=min(1.0, max(0.0, p)), method=method,
        z_statistic=z, notes=notes,
    )


# ---------------------------------------------------------------------------
# multiple-comparison corrections (Holm step-down, BH step-up)


def _sorted_indices(pvals: list[float]) -> list[int]:
    return sorted(range(len(pvals)), key=lambda i: pvals[i])


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm (1979) step-down adjusted p-values, in input order.

    adj_(i) = max_{j<=i} min(1, (m-j+1) * p_(j)) with monotonicity
    enforced from the smallest p upward.  Controls FWER at level alpha:
    reject exactly the hypotheses with adjusted p < alpha.
    """
    for p in pvals:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must be in [0, 1], got {p}")
    m = len(pvals)
    if m == 0:
        return []
    order = _sorted_indices(pvals)
    adj_sorted: list[float] = []
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        running = max(running, min(1.0, (m - rank + 1) * pvals[idx]))
        adj_sorted.append(running)
    out = [0.0] * m
    for pos, idx in enumerate(order):
        out[idx] = adj_sorted[pos]
    return out


def benjamini_hochberg_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg (1995) step-up adjusted p-values, in input order.

    adj_(i) = min_{j>=i} min(1, m/j * p_(j)) with monotonicity enforced
    from the largest p downward.  Controls FALSE DISCOVERY RATE (not
    FWER): use for exploratory dashboards, not confirmatory gates.
    """
    for p in pvals:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must be in [0, 1], got {p}")
    m = len(pvals)
    if m == 0:
        return []
    order = _sorted_indices(pvals)
    adj_sorted: list[float] = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, min(1.0, m * pvals[idx] / rank))
        adj_sorted[rank - 1] = running
    out = [0.0] * m
    for pos, idx in enumerate(order):
        out[idx] = adj_sorted[pos]
    return out


def correct_pvalues(pvals: list[float], method: str) -> dict:
    """Apply a named correction; returns adjusted p-values + metadata."""
    if method == "holm":
        return {
            "method": "holm",
            "controls": "family-wise error rate (FWER); use for gates",
            "adjusted": holm_adjust(pvals),
        }
    if method in ("bh", "benjamini_hochberg"):
        return {
            "method": "benjamini_hochberg",
            "controls": "false discovery rate (FDR); use for dashboards",
            "adjusted": benjamini_hochberg_adjust(pvals),
        }
    if method == "none" or method is None:
        return {
            "method": "none",
            "controls": "nothing (uncorrected; only valid for ONE hypothesis)",
            "adjusted": list(pvals),
        }
    raise ValueError(f"unknown correction: {method!r}")


# ---------------------------------------------------------------------------
# hierarchical bootstrap (scenarios -> tasks; runs held fixed)


def hierarchical_bootstrap(
    data: dict,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    statistic=None,
) -> Estimate:
    """Percentile CI respecting the scenario -> task -> run hierarchy.

    ``data`` maps scenario -> task -> per-task value (float), where the
    value is ALREADY the per-task aggregate over runs (e.g. pass^k, the
    task's mean reward).  Each replicate resamples scenarios with
    replacement, then tasks within each chosen scenario with replacement;
    the per-task values themselves are never resampled ("runs" level held
    fixed).  Resampling rollouts within a task independently reached only
    17% coverage under heterogeneous conditions, while respecting the
    hierarchy restored nominal coverage.  [precipice] section 4.3.

    ``statistic`` receives the replicate's list of sampled per-task
    values (default: their mean).
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not data:
        raise ValueError("data must contain at least one scenario")

    if statistic is None:
        def statistic(values: list[float]) -> float:
            return sum(values) / len(values)

    scenarios = {
        s: {t: (float(v) if not isinstance(v, (list, tuple))
                else sum(v) / len(v))
            for t, v in tasks.items()}
        for s, tasks in data.items()
    }
    empty = [s for s, tasks in scenarios.items() if not tasks]
    if empty:
        raise ValueError(f"scenarios with no tasks: {empty}")

    all_values = [v for tasks in scenarios.values() for v in tasks.values()]
    point = statistic(all_values)

    rng = random.Random(seed)
    scen_names = list(scenarios)
    replicates = []
    for _ in range(n_boot):
        sampled: list[float] = []
        for _s in range(len(scen_names)):
            tasks = scenarios[scen_names[rng.randrange(len(scen_names))]]
            task_names = list(tasks)
            for _t in range(len(task_names)):
                sampled.append(tasks[task_names[rng.randrange(len(task_names))]])
        replicates.append(statistic(sampled))
    replicates.sort()

    def _quantile(q: float) -> float:
        nb = len(replicates)
        idx = min(nb - 1, max(0, int(round(q * (nb - 1)))))
        return replicates[idx]

    lo, hi = _quantile(alpha / 2.0), _quantile(1.0 - alpha / 2.0)
    n_tasks = len(all_values)
    return Estimate(
        value=point,
        ci=(lo, hi),
        method=(
            f"hierarchical percentile bootstrap "
            f"(scenarios={len(scen_names)}, tasks={n_tasks}, "
            f"B={n_boot}, seed={seed})"
        ),
        notes=[
            "scenarios and tasks resampled with replacement; "
            "per-task aggregates (runs level) held fixed during resampling",
        ],
    )


# ---------------------------------------------------------------------------
# Newcombe hybrid-score interval for paired proportion differences


def newcombe_paired(
    a: int, b: int, c: int, d: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Newcombe hybrid-score interval for p1 - p2 with PAIRED data.

    2x2 table of paired binary outcomes (Newcombe 1998, method 10):
        a = both 1,  b = first 0 / second 1,
        c = first 1 / second 0, d = both 0.
    n = a+b+c+d, p1 = (a+b)/n, p2 = (a+c)/n, d_hat = (b-c)/n = p1 - p2.

    The marginal Wilson intervals (l1,u1) and (l2,u2) are combined with
    the observed within-pair correlation phi_hat:

        L = d_hat - sqrt((p1-l1)^2 + (u2-p2)^2 - 2*phi*(p1-l1)*(u2-p2))
        U = d_hat + sqrt((u1-p1)^2 + (p2-l2)^2 - 2*phi*(u1-p1)*(p2-l2))

    with negative radicands snapped to zero.  When phi is undefined (a
    degenerate margin) it is set to 0, widening the interval
    conservatively.  Complements the McNemar test, which alone gives no
    interval.
    """
    n = a + b + c + d
    if n <= 0:
        raise ValueError("need at least one paired observation")
    if min(a, b, c, d) < 0:
        raise ValueError("table counts must be non-negative")
    p1 = (a + b) / n
    p2 = (a + c) / n
    d_hat = p1 - p2

    l1, u1 = wilson_ci(a + b, n, alpha)
    l2, u2 = wilson_ci(a + c, n, alpha)

    phi_den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / phi_den if phi_den > 0 else 0.0
    phi = max(-1.0, min(1.0, phi))

    def _radicand(x: float, y: float) -> float:
        val = x * x + y * y - 2.0 * phi * x * y
        return math.sqrt(max(0.0, val))

    lo = d_hat - _radicand(p1 - l1, u2 - p2)
    hi = d_hat + _radicand(u1 - p1, p2 - l2)
    return (lo, hi)
