# reliagate

**Status: v0.1.0 -- mechanism implemented and offline-tested. NOT validated against real benchmark data yet (see ROADMAP item 1).**

Most agent evaluations run each task once and gate CI on a raw score threshold, so run-to-run noise produces both false regressions and false all-clears. `reliagate` treats the *task* as the unit of inference: run each task `k` times, measure whether it is **consistently** solved, and gate on a paired statistical test between two versions instead of a noisy threshold. It exists because no surveyed tool (Inspect AI, tau2-bench, Langfuse, LangSmith, DeepEval, ...) combines repeated-run reliability metrics, paired significance testing, and an exit-coded CI gate in one package.

## Example (~15 lines)

```python
from reliagate import EvalRun, Run, pass_k, compare, wilson_indicator

base = EvalRun("v123", runs=[
    Run("checkout", 0, True),  Run("checkout", 1, True),  Run("checkout", 2, False),
    Run("refund",   0, True),  Run("refund",   1, None),  Run("refund",   2, True),
    Run("search",   0, False), Run("search",   1, False), Run("search",   2, False),
])

pass_k(base, k=2).value                 # pass^2 = 0.166 (only refund could pass 2/2)
wilson_indicator(base).ci               # Pass^n CI -> (0.0, 0.716) on 1/3 consistent tasks
compare(base_runs, new_runs).p_value    # paired McNemar (exact / mid-p), b-c signed effect
```

And the CI gate:

```bash
reliagate gate --base base.jsonl --new new.jsonl --max-regression 0.05
# exit 0 = pass, exit 2 = significant regression, exit 3 = infra error (never a silent pass)
```

JSONL schema: one trial per line, `{"task_id": "checkout", "attempt": 0, "success": true}` -- `null` means abstained/errored.

## What it computes

| Metric | Definition | Verified against |
|---|---|---|
| `pass^k` | mean over tasks of `C(c,k)/C(n,k)` | brute-force k-subset enumeration; tau-bench arXiv:2406.12045 §3 |
| `pass@k` (unbiased) | mean of `1 - C(n-c,k)/C(n,k)` | brute-force enumeration + the exact Codex Appendix A unbiasedness identity, verified with rational arithmetic (arXiv:2107.03374) |
| `consistency` | consistently / sometimes / never solved | arXiv:2604.17849 §2.2 |
| `wilson_ci` | Wilson score interval | published anchor value 0/10 → [0, 0.2775]; non-degenerate at p̂=0/1 (arXiv:2605.08261 §4.2) |
| bootstrap CI | percentile CI resampling **tasks**, never trials within a task | arXiv:2605.08261 §4.3 (rollout-only bootstrap = 17% coverage) |
| `compare` | McNemar on `z_x = 1[c_x == n]`, exact when discordant < 25 else mid-p | hand-computed exact/mid-p values, incl. the b=5,c=0 disagreement case (Fagerland et al. 2013) |
| `required_runs` / `plan` | simulation-based task-count planning under documented assumptions | deterministic seeded sims; power grows as effect shrinks |
| `sprt_stop` | Wald SPRT over the per-task indicator stream; capped runs are INCONCLUSIVE, never a pass | Wald boundaries `ln((1-β)/α)`, `ln(β/(1-α))` |

## Design decisions you should know

- **Abstentions are never failures.** `success=None` trials count in the denominator `n`, are excluded from successes `c` (the conservative direction), and every report states this treatment explicitly. This closes the ambiguity tau2-bench hit (issue #497): the convention cannot silently change task weights.
- **`consistent` requires decisive evidence.** A task with an abstained trial cannot be `consistent`, because consistency claims all-n success.
- **Zero dependencies.** The core is pure stdlib (`math.comb`, `math.erf`, `random.Random` with fixed seeds). Tests run with plain `python3 -m unittest` -- no numpy, no scipy, no pytest.
- **Exact integer combinatorics.** Estimators use `math.comb`, matching the papers' closed forms without float accumulation.

## Live runs (opt-in)

All live agent runs of this library use the **Z.ai Coding Plan** endpoint, OpenAI-compatible:

- base URL: `https://api.z.ai/api/coding/paas/v4`
- model: `glm-5.3-flash` (pinned -- the only supported model)
- key: `RELIAGATE_ZAI_KEY` environment variable

The live path (`reliagate.adapters.runner.make_zai_agent_fn`) is strictly opt-in behind that env var; nothing in the library touches the network without it, and all statistics + tests are offline and key-free. On any billing or endpoint error the runner **stops and raises** -- it never falls back to another provider or to pay-per-call endpoints.

## Limitations (honest)

- **Single-model pair comparisons only.** `compare` tests two versions of one agent on one task set; it does not rank many agents.
- **Per-task binary outcomes.** Trials are pass/fail/abstain. Partial scores (`Trial.score` in the source research) are not yet modeled.
- **Small-sample caveats.** The exact McNemar test is conservative; mid-p is closer to nominal but neither is magic. Below ~10 tasks the gate's "not significant" often means "underpowered" -- use `plan()` to size the study first. Bootstrap CIs with <20 tasks are wide by construction.
- **Planning assumptions are simple** (independent Bernoulli indicators); real paired runs share task difficulty, which usually makes real power *better* than simulated, but this is unverified against real data so far.
- **No hierarchical bootstrap yet.** Tasks nesting in suites/categories (arXiv:2605.08261 §4.3) are on the ROADMAP.
- Not benchmark-reproduced. All validation so far is offline property testing; ROADMAP item 1 is the first real-data run.

## Install / test

```bash
git clone https://github.com/Gjusev/reliagate && cd reliagate
python3 -m unittest          # 90 tests, zero dependencies required
pip install .                # optional; provides the `reliagate` command
```

See [ROADMAP.md](ROADMAP.md) for what's next and [CONTINUE.md](CONTINUE.md) if you are a coding agent picking this up.

## References

- Yao et al., τ-bench, arXiv:2406.12045 (pass^k, §3)
- Chen et al., Codex/HumanEval, arXiv:2107.03374 (unbiased pass@k, §2.1 + App. A)
- *On the Reliability of Computer Use Agents*, arXiv:2604.17849 (§2.2 consistency classes, McNemar on z_x)
- *Computer Use at the Edge of the Statistical Precipice*, arXiv:2605.08261 (§4.2 Wilson, §4.3 task-level bootstrap)
- Fagerland, Lydersen & Laake 2013, Stat Med 32:3086 (exact vs mid-p McNemar)
- tau2-bench issue #497 (infra-error convention ambiguity)

MIT -- Youssef Ouhaghi, 2026.
