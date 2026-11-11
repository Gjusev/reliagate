"""reliagate -- reliability metrics and CI gates for AI agents."""

from .model import (
    EvalRun,
    Run,
    abstention_report,
    consistency_class,
    consistency_classes,
    group_by_task,
    task_counts,
)
from .metrics import (
    Estimate,
    bootstrap_pass_pow_k,
    pass_at_k,
    pass_k,
    pass_pow_k,
    per_task,
    consistency,
    wilson_ci,
    wilson_indicator,
)
from .compare import (
    ComparisonResult,
    compare,
    mcnemar_exact,
    mcnemar_midp,
)
from .planning import (
    PlanResult,
    SprtResult,
    consistency_settled,
    plan,
    required_runs,
    sprt_stop,
)
from .gate import (
    EXIT_ERROR,
    EXIT_PASS,
    EXIT_REGRESSION,
    GateReport,
    gate,
)
from .adapters.jsonl import JSONLError, read_jsonl, write_jsonl

__version__ = "0.1.0"

__all__ = [
    "EvalRun",
    "Run",
    "abstention_report",
    "consistency_class",
    "consistency_classes",
    "group_by_task",
    "task_counts",
    "Estimate",
    "bootstrap_pass_pow_k",
    "pass_at_k",
    "pass_k",
    "pass_pow_k",
    "per_task",
    "consistency",
    "wilson_ci",
    "wilson_indicator",
    "ComparisonResult",
    "compare",
    "mcnemar_exact",
    "mcnemar_midp",
    "PlanResult",
    "SprtResult",
    "consistency_settled",
    "plan",
    "required_runs",
    "sprt_stop",
    "EXIT_ERROR",
    "EXIT_PASS",
    "EXIT_REGRESSION",
    "GateReport",
    "gate",
    "JSONLError",
    "read_jsonl",
    "write_jsonl",
]
