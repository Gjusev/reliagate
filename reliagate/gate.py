"""CI gate: exits non-zero on a statistically significant regression.

Exit codes (explicit, so CI never guesses):
    0  gate passed -- the drop in consistently-solved tasks is within
       --max-regression
    2  significant statistical regression: the drop exceeds --max-regression
       AND the paired McNemar (auto exact/mid-p) p-value is below alpha
    3  infra/usage error -- unreadable files, malformed JSONL, mismatched
       task sets or trial counts, argument errors, or more abstained trials
       than --max-abstention allows.  NEVER silently passes: any doubt is 3.
    4  inconclusive: the drop exceeds --max-regression but is not
       significant.  "Not significant" is not "safe": the study is too small
       to tell, so the gate neither passes nor claims a regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .compare import ComparisonResult, compare
from .model import EvalRun, abstention_report


EXIT_PASS = 0
EXIT_REGRESSION = 2
EXIT_ERROR = 3
EXIT_INCONCLUSIVE = 4


@dataclass
class GateReport:
    decision: str                 # "pass" | "regression" | "inconclusive" | "error"
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
    max_abstention: float = 0.05,
) -> GateReport:
    """Run the gate.  Raises nothing on data problems -- returns exit 3.

    ``max_abstention`` caps the share of abstained/infra-errored trials in
    either version.  Abstentions count in n and never as successes, so a
    version with more infra errors looks less consistent; above the cap the
    verdict would measure the infrastructure, not the agent, and the gate
    returns 3 instead.
    """
    if not 0 <= max_regression <= 1:
        return GateReport(
            "error", EXIT_ERROR,
            f"--max-regression must be in [0, 1], got {max_regression}",
        )
    if not 0 < alpha < 1:
        return GateReport(
            "error", EXIT_ERROR, f"alpha must be in (0, 1), got {alpha}",
        )
    if not 0 <= max_abstention <= 1:
        return GateReport(
            "error", EXIT_ERROR,
            f"--max-abstention must be in [0, 1], got {max_abstention}",
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

    rates = {
        side: (a["trials_abstained"] / a["trials_total"] if a["trials_total"] else 0.0)
        for side, a in abstention.items()
    }
    worst = max(rates, key=rates.get)
    if rates[worst] > max_abstention:
        report.decision = "error"
        report.exit_code = EXIT_ERROR
        report.reason = (
            f"{worst} has {rates[worst]:.1%} abstained/infra-errored trials "
            f"(> max-abstention {max_abstention:.1%}); a verdict would measure "
            "the infrastructure, not the agent -- fix the runs or raise "
            "--max-abstention deliberately"
        )
        return report

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
        report.decision = "inconclusive"
        report.exit_code = EXIT_INCONCLUSIVE
        report.reason = (
            f"drop {drop:.4f} exceeds max-regression but is NOT statistically "
            f"significant ({cmp.test_used} p={cmp.p_value:.4g} >= alpha={alpha}); "
            "inconclusive -- the study is too small to tell; size it with plan()"
        )
    elif cmp.p_value < alpha and cmp.b_minus_c < 0:
        report.reason = (
            f"significant negative shift (p={cmp.p_value:.4g}) but within "
            f"max-regression {max_regression}; gate passes"
        )
    return report
