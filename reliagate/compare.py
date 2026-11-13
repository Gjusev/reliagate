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

Companion statistics (computed alongside every McNemar):

* ``wilcoxon`` -- Wilcoxon signed-rank on d_x = c_x_new - c_x_base
  (reliability paper section 2.2): catches partial consistency shifts
  (e.g. 1 -> 2 of 3) that the binary indicator discards.
* ``newcombe`` -- Newcombe hybrid-score interval on the difference of
  the two consistency proportions (McNemar alone yields no interval).
* ``task_categories`` + ``correction`` -- per-category McNemar tests
  with Holm (gates) or Benjamini-Hochberg (dashboards) adjustment over
  the family.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .model import EvalRun, consistency_classes, group_by_task, task_counts
from .stats import correct_pvalues, newcombe_paired, wilcoxon_signed_rank


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
    # companion statistics (all None-safe: present only when computed)
    wilcoxon: Optional[dict] = None       # signed-rank on d_x = c_new - c_base
    newcombe: Optional[dict] = None       # hybrid-score interval on p1 - p2
    category_tests: list[dict] = field(default_factory=list)
    correction: Optional[dict] = None     # multiplicity correction metadata

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


def _wilcoxon_dict(base_counts: dict, new_counts: dict, task_ids) -> Optional[dict]:
    """Signed-rank on paired d_x = c_x_new - c_x_base (arXiv:2604.17849 2.2)."""
    diffs = [new_counts[t][0] - base_counts[t][0] for t in task_ids]
    r = wilcoxon_signed_rank(diffs)
    return {
        "n_pairs": r.n_pairs,
        "n_nonzero": r.n_nonzero,
        "w": r.w,
        "w_plus": r.w_plus,
        "w_minus": r.w_minus,
        "p_value": r.p_value,
        "method": r.method,
        "z_statistic": r.z_statistic,
        "notes": r.notes,
    }


def _newcombe_dict(base_classes: dict, new_classes: dict) -> dict:
    """Newcombe hybrid-score interval on the consistency-indicator difference."""
    a = sum(1 for t in base_classes
            if base_classes[t] == "consistent" and new_classes[t] == "consistent")
    b = sum(1 for t in base_classes
            if base_classes[t] != "consistent" and new_classes[t] == "consistent")
    c = sum(1 for t in base_classes
            if base_classes[t] == "consistent" and new_classes[t] != "consistent")
    d = sum(1 for t in base_classes
            if base_classes[t] != "consistent" and new_classes[t] != "consistent")
    n = a + b + c + d
    lo, hi = newcombe_paired(a, b, c, d)
    return {
        "table": {"both": a, "gained": b, "lost": c, "neither": d},
        "difference": (b - c) / n,
        "ci": (lo, hi),
        "method": "newcombe hybrid score (paired)",
    }


def compare(
    base: EvalRun,
    new: EvalRun,
    test: str = "auto",
    correction: Optional[str] = None,
    task_categories: Optional[dict] = None,
) -> ComparisonResult:
    """Paired McNemar on per-task 'consistently solved' indicators.

    Companion statistics computed alongside the McNemar p-value:

    * ``wilcoxon`` -- signed-rank on the paired per-task success-count
      differences d_x = c_x_new - c_x_base (catches partial consistency
      shifts that the binary indicator discards).
    * ``newcombe`` -- hybrid-score interval on the difference of the
      consistency proportions (McNemar alone yields no interval).

    With ``task_categories`` ({task_id: category}) a per-category McNemar
    is run for each category, and ``correction`` ("holm" for gates, "bh"
    for dashboards) adjusts the family of category p-values.  Without a
    family, ``correction`` is ignored.
    """
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

    wil = _wilcoxon_dict(base_counts, new_counts, base_classes.keys())
    nc = _newcombe_dict(base_classes, new_classes)

    category_tests: list[dict] = []
    correction_report: Optional[dict] = None
    if task_categories:
        missing = sorted(set(base_classes) - set(task_categories))
        if missing:
            raise ValueError(
                f"task_categories missing entries for tasks: {missing}"
            )
        categories = sorted(set(task_categories[t] for t in base_classes))
        for cat in categories:
            cat_tasks = [t for t in base_classes if task_categories[t] == cat]
            cat_base = EvalRun(
                name=f"{base.name}:{cat}",
                runs=[r for r in base.runs if r.task_id in set(cat_tasks)],
            )
            cat_new = EvalRun(
                name=f"{new.name}:{cat}",
                runs=[r for r in new.runs if r.task_id in set(cat_tasks)],
            )
            sub = compare(cat_base, cat_new, test=test)
            category_tests.append({
                "category": cat,
                "n_tasks": sub.n_tasks,
                "b": sub.b,
                "c": sub.c,
                "p_value": sub.p_value,
                "test_used": sub.test_used,
                "newcombe": sub.newcombe,
                "adjusted_p_value": None,
            })
        if correction is not None:
            family = correct_pvalues(
                [ct["p_value"] for ct in category_tests], correction
            )
            for ct, adj in zip(category_tests, family["adjusted"]):
                ct["adjusted_p_value"] = adj
            correction_report = {
                "method": family["method"],
                "controls": family["controls"],
                "n_hypotheses": len(category_tests),
            }

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
        wilcoxon=wil,
        newcombe=nc,
        category_tests=category_tests,
        correction=correction_report,
    )
