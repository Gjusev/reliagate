"""CI gate: exits non-zero on a statistically significant regression.

Exit codes (explicit, so CI never guesses):
    0  gate passed -- no significant regression beyond --max-regression
    2  significant statistical regression detected
    3  infra/usage error -- unreadable files, malformed JSONL, mismatched
       task sets, argument errors.  NEVER silently passes: any doubt is 3.

Regression rule: the gate fails (2) only when the drop in the fraction of
consistently-solved tasks exceeds --max-regression AND the paired McNemar
(auto exact/mid-p) p-value is below alpha.  A tiny-but-significant dip
within the allowed margin passes; a big-but-noisy dip is reported as not
significant, not as a regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .compare import ComparisonResult, compare
from .model import EvalRun, abstention_report


EXIT_PASS = 0
EXIT_REGRESSION = 2
EXIT_ERROR = 3


@dataclass
class GateReport:
    decision: str                 # "pass" | "regression" | "error"
    exit_code: int
    reason: str
    base_pass_pow_n: Optional[float] = None
    new_pass_pow_n: Optional[float] = None
    max_regression: Optional[float] = None
    alpha: Optional[float] = None
    comparison: Optional[ComparisonResult] = None
    abstention: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "decision": self.decision,
            "exit_code": self.exit_code,
            "reason": self.reason,
            "base_pass_pow_n": self.base_pass_pow_n,
            "new_pass_pow_n": self.new_pass_pow_n,
            "max_regression": self.max_regression,
            "alpha": self.alpha,
            "abstention": self.abstention,
        }
        if self.comparison is not None:
            cmp = self.comparison
            d["comparison"] = {
                "b_gained": cmp.b,
                "c_lost": cmp.c,
                "b_minus_c": cmp.b_minus_c,
                "p_value": cmp.p_value,
                "test_used": cmp.test_used,
                "n_tasks": cmp.n_tasks,
                "class_changes": cmp.class_changes,
                "notes": cmp.notes,
            }
        return d


def gate(
    base: EvalRun,
    new: EvalRun,
    max_regression: float = 0.05,
    alpha: float = 0.05,
) -> GateReport:
    """Run the gate.  Raises nothing on data problems -- returns exit 3."""
    if not 0 <= max_regression <= 1:
        return GateReport(
            "error", EXIT_ERROR,
            f"--max-regression must be in [0, 1], got {max_regression}",
        )
    if not 0 < alpha < 1:
        return GateReport(
            "error", EXIT_ERROR, f"alpha must be in (0, 1), got {alpha}",
        )

    try:
        cmp = compare(base, new)
    except ValueError as exc:
        return GateReport("error", EXIT_ERROR, f"invalid input: {exc}",
                          abstention={
                              "base": abstention_report(base),
                              "new": abstention_report(new),
                          })

    drop = cmp.base_pass_pow_n - cmp.new_pass_pow_n
    abstention = {
        "base": abstention_report(base),
        "new": abstention_report(new),
    }
    report = GateReport(
        decision="pass",
        exit_code=EXIT_PASS,
        reason="no significant regression",
        base_pass_pow_n=cmp.base_pass_pow_n,
        new_pass_pow_n=cmp.new_pass_pow_n,
        max_regression=max_regression,
        alpha=alpha,
        comparison=cmp,
        abstention=abstention,
    )

    if drop > max_regression and cmp.p_value < alpha:
        report.decision = "regression"
        report.exit_code = EXIT_REGRESSION
        report.reason = (
            f"consistently-solved tasks dropped by {drop:.4f} "
            f"(> max-regression {max_regression}) and the drop is "
            f"significant ({cmp.test_used} p={cmp.p_value:.4g} < alpha={alpha}); "
            f"b-c={cmp.b_minus_c}"
        )
    elif drop > max_regression:
        report.reason = (
            f"drop {drop:.4f} exceeds max-regression but is NOT statistically "
            f"significant ({cmp.test_used} p={cmp.p_value:.4g} >= alpha={alpha}); "
            "gate passes -- widen the study before trusting this"
        )
    elif cmp.p_value < alpha and cmp.b_minus_c < 0:
        report.reason = (
            f"significant negative shift (p={cmp.p_value:.4g}) but within "
            f"max-regression {max_regression}; gate passes"
        )
    return report
