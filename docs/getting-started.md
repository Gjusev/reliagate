# Getting started

## Install

reliagate is pure Python (3.10+) with zero runtime dependencies.

```bash
pip install git+https://github.com/Gjusev/reliagate   # provides the `reliagate` command
```

Or work from a clone. Nothing needs installing to run the tests:

```bash
git clone https://github.com/Gjusev/reliagate && cd reliagate
python -m unittest discover        # 148 tests, offline, stdlib only
python -m reliagate --version
```

## The data format

One JSON object per line, one line per **trial** (one execution of one task):

```json
{"task_id": "checkout", "attempt": 0, "success": true}
{"task_id": "checkout", "attempt": 1, "success": false}
{"task_id": "checkout", "attempt": 2, "success": null}
```

| field | type | meaning |
|---|---|---|
| `task_id` | string | the task; the unit of inference |
| `attempt` | int, 0-based | which repetition of the task this is; `(task_id, attempt)` must be unique |
| `success` | `true` / `false` / `null` | `null` = abstained or infra error: counted in `n`, never as a failure |
| `version` | string, optional | free-form label, e.g. a commit SHA |
| `metadata` | object, optional | anything you want to carry along |

`read_jsonl` validates strictly and raises `JSONLError` on malformed input. The CLI turns that into exit code `3`.

## Walkthrough: the example pair

[`examples/base.jsonl`](../examples/base.jsonl) and [`examples/new.jsonl`](../examples/new.jsonl) are the same 40 tasks run 4 times each, before (`v1`) and after (`v2`) a change. In `v2`, nine tasks that were solved on every run now fail some of the time.

```python
from reliagate import read_jsonl, pass_k, pass_at_k, consistency, compare

v1 = read_jsonl("examples/base.jsonl")
v2 = read_jsonl("examples/new.jsonl")

pass_at_k(v1, k=4), pass_at_k(v2, k=4)          # (1.0, 1.0)   at least once: no change
pass_k(v1, k=4).value, pass_k(v2, k=4).value    # (0.85, 0.65) every time: a real drop

consistency(v2)["counts"]
# {'consistent': 26, 'sometimes': 14, 'never': 0, 'no_trials': 0}

r = compare(v1, v2)
r.b, r.c, r.test_used, r.p_value   # (1, 9, 'mcnemar_exact', 0.02148...)
r.newcombe["ci"]                   # (-0.341, -0.055): the drop, with an interval
r.class_changes[:2]                # [{'task_id': 'task-00', 'base': 'consistent', 'new': 'sometimes'}, ...]
```

The same comparison from the shell:

```bash
reliagate compare --base examples/base.jsonl --new examples/new.jsonl
reliagate gate    --base examples/base.jsonl --new examples/new.jsonl ; echo "exit $?"   # exit 2
```

## Collecting runs from your own agent

Any function `task_id -> True | False | None` works. `run_agent` runs it `k` times per task on a thread pool, with an optional cost budget:

```python
from reliagate.adapters.runner import run_agent
from reliagate import write_jsonl

def my_agent(task_id: str) -> bool | None:
    ...  # run your agent on the task, return whether it succeeded (None if it could not run)

report = run_agent(my_agent, tasks=["checkout", "refund", "search"], k=4,
                   max_workers=8, cost_budget=5.0, estimated_cost_per_call=0.01, version="pr-123")
write_jsonl(report.run, "evals/pr.jsonl")
```

Attempts the budget did not allow are recorded as `success=None` with `metadata={"reason": "budget_exhausted"}`. They count as abstentions, never as failures.

If you already have results in another format, write the JSONL yourself. It's one line per trial.

## Next

- [Metrics](metrics.md): what each number means and how it is verified
- [CI gate](ci-gate.md): wiring the gate into GitHub Actions
- [Design and limitations](design.md)
