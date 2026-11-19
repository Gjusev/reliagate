<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Gjusev/reliagate/main/assets/brand/lockup-dark.svg">
    <img src="https://raw.githubusercontent.com/Gjusev/reliagate/main/assets/brand/lockup-light.svg" alt="reliagate" width="420">
  </picture>
</p>

<p align="center">
  <b>Your agent passed CI once. That proves nothing.</b><br>
  Reliability metrics and statistically sound CI gates for AI agents evaluated with repeated runs.
</p>

<p align="center">
  <a href="https://github.com/Gjusev/reliagate/actions/workflows/tests.yml"><img src="https://github.com/Gjusev/reliagate/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/dependencies-0-2EBD78" alt="zero dependencies">
  <img src="https://img.shields.io/badge/python-3.10%2B-2EBD78" alt="python 3.10+">
  <a href="https://github.com/Gjusev/reliagate/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-2EBD78" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://github.com/Gjusev/reliagate/blob/main/assets/video/reliagate.mp4"><img src="https://raw.githubusercontent.com/Gjusev/reliagate/main/assets/video/preview.gif" alt="reliagate in 30 seconds: one run passes, four runs don't; pass@4 stays 1.00 while pass^4 drops 0.85 to 0.65; the gate exits 2" width="760"></a>
  <br>
  <sub><a href="https://github.com/Gjusev/reliagate/blob/main/assets/video/reliagate.mp4">Watch the 30-second film</a>. Every number in it is real output of this repo's <a href="https://github.com/Gjusev/reliagate/blob/main/examples/">example files</a>.</sub>
</p>

---

Most agent evals run each task **once** and gate CI on a score threshold. Agents are stochastic, so the same commit can pass on Monday and fail on Tuesday, and you get false alarms and false all-clears.

**reliagate** treats the *task* as the unit of evidence:

1. **Run every task `k` times** and ask whether it is solved *consistently*, not just once.
2. **Measure reliability**, not luck: `pass^k`, unbiased `pass@k`, consistency classes, Wilson and task-level bootstrap intervals.
3. **Gate on a paired statistical test** between two versions (McNemar, with Wilcoxon and a Newcombe interval alongside), with exit codes your CI can trust.

Pure Python, **zero dependencies**, 148 offline tests.

## Why pass@k is not enough

The film above uses the two files in [`examples/`](examples/): the same 40 tasks, 4 runs each, before and after a change.

| | v1 | v2 |
|---|---|---|
| `pass@4`: solved at least once | **1.00** | **1.00** |
| `pass^4`: solved every time | **0.85** | **0.65** |

`pass@4` says nothing changed. In fact 9 tasks that used to pass every run now fail some of the time, and only 1 went the other way:

```console
$ reliagate gate --base examples/base.jsonl --new examples/new.jsonl
gate decision: REGRESSION (exit 2)
  consistently solved: base 34/40 -> new 26/40
  gained (b)=1  lost (c)=9  b-c=-8  test=mcnemar_exact  p=0.02148
  ...
```

## Quickstart

```bash
pip install reliagate
```

```python
from reliagate import read_jsonl, pass_k, pass_at_k, compare, gate

v1 = read_jsonl("examples/base.jsonl")   # one line per trial: {"task_id", "attempt", "success"}
v2 = read_jsonl("examples/new.jsonl")

pass_at_k(v2, k=4)                        # 1.0   solved at least once in 4
pass_k(v2, k=4).value                     # 0.65  solved in all 4
pass_k(v2, k=4, ci="wilson_indicator").ci # (0.495, 0.779)

r = compare(v1, v2)                       # paired McNemar on "consistently solved"
r.b, r.c, r.p_value                       # (1, 9, 0.02148)

gate(v1, v2, max_regression=0.05).exit_code   # 2: significant regression
```

Trials are `{"task_id": "checkout", "attempt": 0, "success": true}`. `success: null` means the trial abstained or hit an infrastructure error. It stays in the denominator and is **never** counted as a failure.

## Gate your CI

```yaml
- run: pip install reliagate
- run: reliagate gate --base evals/main.jsonl --new evals/pr.jsonl --max-regression 0.05
```

| Exit | Meaning |
|---|---|
| `0` | pass: the drop in consistently-solved tasks is within `--max-regression` |
| `2` | significant regression in consistently-solved tasks |
| `3` | infra or usage error, including too many infra-errored trials: never a silent pass |
| `4` | inconclusive: a drop beyond the limit that the data can't confirm. Run more tasks |

Full recipe, including how to size the run with `plan()` and stop early with SPRT: [docs/ci-gate.md](docs/ci-gate.md).

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | data format, the example walkthrough, collecting runs |
| [Metrics](docs/metrics.md) | every estimator, its definition and what it is verified against |
| [CI gate](docs/ci-gate.md) | GitHub Actions recipe, exit codes, planning and early stopping |
| [Design and limitations](docs/design.md) | abstentions, zero dependencies, what reliagate does not do |
| [First real study](docs/results/first-study.md) | the toolkit on 494 real tau-bench retail tasks |
| [Changelog](CHANGELOG.md) | what changed in each release |
| [Roadmap](ROADMAP.md) | what's done and what's next |

## Validated on real data

[docs/results/first-study.md](docs/results/first-study.md) applies reliagate to real tau-bench retail trajectories, regenerated deterministically by `scripts/first_study.py` with no LLM calls. On 494 tasks at k=2, **pass^2 = 0.907** (Wilson 95% CI [0.878, 0.930]). With k=4, the pass@k minus pass^k gap reaches **+0.20** for one gating policy. All five paired policy comparisons come out null and underpowered, and the study says so instead of rounding them into a win.

## Contributing

Issues and PRs are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first: the core stays stdlib-only and every claim must be backed by a test or a rerunnable script. If you use reliagate in research, please cite it ([CITATION.cff](CITATION.cff)).

MIT © 2026 Youssef Ouhaghi
