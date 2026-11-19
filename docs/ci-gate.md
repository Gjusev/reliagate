# The CI gate

```bash
reliagate gate --base base.jsonl --new new.jsonl [--max-regression 0.05] [--alpha 0.05] [--max-abstention 0.05] [--json]
```

The gate looks at the drop in the share of consistently-solved tasks:

- within `--max-regression`: **pass** (0)
- beyond it, and the paired McNemar p-value is below `--alpha`: **regression** (2)
- beyond it, but not significant: **inconclusive** (4). "Not significant" is not "safe": the study is too small to tell. Size it with `plan()` and rerun.

Before judging, the gate checks the data: both versions need the same tasks with the same number of trials each, and neither may have more than `--max-abstention` of its trials abstained or infra-errored. Otherwise the verdict would measure the data or the infrastructure, not the agent, so the gate returns 3.

## Exit codes are a contract

| Exit | Decision | When |
|---|---|---|
| `0` | `pass` | the drop is within `--max-regression` |
| `2` | `regression` | significant drop beyond `--max-regression` |
| `3` | `error` | unreadable or malformed input, mismatched tasks or trial counts, too many abstained trials, invalid arguments |
| `4` | `inconclusive` | a drop beyond `--max-regression` that is not significant |

`2` always means a statistical regression and nothing else. Argument errors exit `3`, not argparse's default `2`. Infra failures never exit `0`. Whether `4` blocks your merge is your call; most teams should treat it as a failure until the study is big enough.

## GitHub Actions

```yaml
name: agent-eval
on: pull_request

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install git+https://github.com/Gjusev/reliagate

      # Produce evals/pr.jsonl with your own harness: every task, k runs each.
      - run: python evals/run.py --k 4 --out evals/pr.jsonl

      # evals/main.jsonl: the baseline, committed or downloaded from the main branch's last run.
      # Exit 4 (inconclusive) fails the step too; allow it with `|| [ $? -eq 4 ]` if you prefer.
      - run: reliagate gate --base evals/main.jsonl --new evals/pr.jsonl --max-regression 0.05
```

Add `--json` to get a machine-readable report (decision, p-value, class changes, abstention counts) for a PR comment.

## How many tasks and runs?

The gate is honest about power: below ~10 tasks, "not significant" usually means "too small to tell". Size the run before trusting it:

```python
from reliagate import read_jsonl, plan

pilot = read_jsonl("evals/main.jsonl")
p = plan(pilot, effect=0.10)      # detect a 10-point drop in consistently-solved tasks
p.n_tasks, p.achieved_power       # e.g. (153, 0.805) for the example pilot
p.assumptions                     # stated, not hidden
```

The more runs per task (`k`), the stricter "consistent" gets. `k = 3` to `5` is a common range.

## Stopping early

For expensive agents, `sprt_stop` evaluates the candidate task by task and stops once the evidence crosses a Wald boundary:

```python
from reliagate import sprt_stop

res = sprt_stop(base, candidate_so_far, p1=0.75)
res.decision   # "accept" | "reject" | "continue"
```

`continue` at the task cap is **inconclusive**. Treat it as "run more", never as a pass.

## Reading the report

```text
gate decision: REGRESSION (exit 2)
  consistently solved: base 34/40 -> new 26/40
  gained (b)=1  lost (c)=9  b-c=-8  test=mcnemar_exact  p=0.02148
  class change: task-00: consistent -> sometimes
  ...
  abstentions: base 0/160 trials, new 0/160 trials (counted in n, never counted as failure)
```

`b` and `c` are the discordant tasks. Only those carry evidence about the change. `class change` lines tell you exactly which tasks to look at.
