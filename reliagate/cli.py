"""reliagate command-line interface.

Subcommands:
    gate      CI gate over two JSONL trial files.
    compare   human-readable comparison report, no exit code semantics.

Exit codes for `gate` (see reliagate.gate):
    0 pass -- 2 significant regression -- 3 infra/usage error.
Argument/usage errors also exit 3, never argparse's default 2, so "2"
always means exactly one thing: statistical regression.
"""

from __future__ import annotations

import argparse
import json
import sys

from .adapters.jsonl import JSONLError, read_jsonl
from .compare import compare
from .gate import EXIT_ERROR, EXIT_PASS, EXIT_REGRESSION, gate


class _ArgErrorParser(argparse.ArgumentParser):
    """argparse that exits 3 (infra/usage) instead of the default 2."""

    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR)


def _build_parser() -> _ArgErrorParser:
    parser = _ArgErrorParser(
        prog="reliagate",
        description="reliability metrics and CI gates for AI agents",
    )
    parser.add_argument(
        "--version", action="store_true", help="print version and exit"
    )
    sub = parser.add_subparsers(dest="command")

    g = sub.add_parser(
        "gate",
        help="CI gate: exit 0 pass / 2 significant regression / 3 infra error",
    )
    g.add_argument("--base", required=True, help="base version trials (JSONL)")
    g.add_argument("--new", required=True, help="candidate trials (JSONL)")
    g.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="max tolerated drop in consistently-solved task fraction "
        "(default 0.05)",
    )
    g.add_argument(
        "--alpha", type=float, default=0.05, help="significance level (default 0.05)"
    )
    g.add_argument(
        "--json", action="store_true", help="print the full report as JSON"
    )

    c = sub.add_parser("compare", help="human-readable paired comparison")
    c.add_argument("--base", required=True)
    c.add_argument("--new", required=True)
    return parser


def _load(path: str):
    try:
        return read_jsonl(path)
    except JSONLError as exc:
        print(f"reliagate: infra error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(f"reliagate {__version__}")
        return EXIT_PASS
    if args.command == "gate":
        base = _load(args.base)
        new = _load(args.new)
        report = gate(base, new, max_regression=args.max_regression, alpha=args.alpha)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            cmp = report.comparison
            print(
                f"gate decision: {report.decision.upper()} "
                f"(exit {report.exit_code})"
            )
            if cmp is not None:
                print(
                    f"  consistently solved: base {cmp.base_consistent}/"
                    f"{cmp.n_tasks} -> new {cmp.new_consistent}/{cmp.n_tasks}"
                )
                print(
                    f"  gained (b)={cmp.b}  lost (c)={cmp.c}  b-c={cmp.b_minus_c}  "
                    f"test={cmp.test_used}  p={cmp.p_value:.4g}"
                )
                for change in cmp.class_changes:
                    print(
                        f"  class change: {change['task_id']}: "
                        f"{change['base']} -> {change['new']}"
                    )
            print(f"  reason: {report.reason}")
            ab = report.abstention
            print(
                f"  abstentions: base {ab['base']['trials_abstained']}/"
                f"{ab['base']['trials_total']} trials, "
                f"new {ab['new']['trials_abstained']}/{ab['new']['trials_total']} "
                f"trials (counted in n, never counted as failure)"
            )
        raise SystemExit(report.exit_code)
    if args.command == "compare":
        base = _load(args.base)
        new = _load(args.new)
        cmp = compare(base, new)
        print(f"tasks: {cmp.n_tasks}")
        print(f"b (gained consistent)={cmp.b}  c (lost)={cmp.c}  b-c={cmp.b_minus_c}")
        print(f"test={cmp.test_used}  p={cmp.p_value:.4g}")
        for change in cmp.class_changes:
            print(
                f"class change: {change['task_id']}: "
                f"{change['base']} -> {change['new']}"
            )
        return EXIT_PASS
    parser.print_help()
    return EXIT_PASS


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
