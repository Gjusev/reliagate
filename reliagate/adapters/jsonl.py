"""JSONL in/out adapter (stdlib only -- always available).

Line schema (one trial per line):
    {"task_id": "checkout", "attempt": 0, "success": true,
     "version": "abc123", "metadata": {"latency_ms": 4311}}
``success`` is true, false or null (null = abstained/infra error).
"""

from __future__ import annotations

import json
from typing import Any

from ..model import EvalRun, Run


class JSONLError(ValueError):
    """Raised on malformed JSONL -- the gate maps this to exit 3."""


def write_jsonl(run: EvalRun, path: str) -> int:
    """Write an EvalRun to path; returns the number of lines."""
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in run.runs:
            fh.write(json.dumps(r.to_dict() if hasattr(r, "to_dict") else {
                "task_id": r.task_id,
                "attempt": r.attempt,
                "success": r.success,
                "version": r.version,
                "metadata": r.metadata,
            }) + "\n")
            count += 1
    return count


def read_jsonl(path: str, name: str | None = None) -> EvalRun:
    """Read a JSONL file of trials into an EvalRun.

    Raises JSONLError (a ValueError) on unreadable files or malformed
    lines -- callers must surface this as an infra error, never as a
    comparison result.
    """
    runs: list[Run] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        raise JSONLError(f"cannot read {path!r}: {exc}") from exc

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JSONLError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise JSONLError(f"{path}:{lineno}: expected a JSON object")
        missing = [k for k in ("task_id", "attempt", "success") if k not in obj]
        if missing:
            raise JSONLError(
                f"{path}:{lineno}: missing required keys {missing}"
            )
        metadata = obj.get("metadata", {})
        if not isinstance(metadata, dict):
            raise JSONLError(f"{path}:{lineno}: 'metadata' must be an object")
        try:
            runs.append(
                Run(
                    task_id=obj["task_id"],
                    attempt=obj["attempt"],
                    success=obj["success"],
                    version=obj.get("version"),
                    metadata=metadata,
                )
            )
        except (TypeError, ValueError) as exc:
            raise JSONLError(f"{path}:{lineno}: {exc}") from exc

    if not runs:
        raise JSONLError(f"{path}: contains no trials")
    return EvalRun(name=name or path, runs=runs)
