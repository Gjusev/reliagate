# CONTINUE.md -- handoff for coding agents

## Where this stopped

v0.1.0 is committed and pushed. The core library is complete and offline-tested:

- `reliagate/model.py` -- Run/EvalRun, abstention accounting (None = abstained, counted in n, never a failure)
- `reliagate/metrics.py` -- pass^k, unbiased pass@k, consistency classes, Wilson CI, task-level bootstrap, rule-of-three
- `reliagate/compare.py` -- McNemar exact (< 25 discordant) / mid-p (>= 25), class-change table
- `reliagate/planning.py` -- `required_runs`/`plan` (seeded simulation), `sprt_stop`, `consistency_settled`
- `reliagate/gate.py` + `reliagate/cli.py` -- gate CLI with exit codes 0/2/3
- `reliagate/adapters/` -- JSONL (stdlib), nitelos (lazy mapping), runner with concurrency + cost budget and the opt-in live Z.ai path
- `tests/` -- 90 tests, all green with `python3 -m unittest`

Everything is validated against paper formulas and brute-force enumeration -- NOT against real benchmark data. ROADMAP item 1 is the next job.

## Next steps (in order)

1. **ROADMAP item 1**: pull tau-bench retail trajectories from `Gjusev/gate-calibrated`, convert to JSONL, run the first real study, add `examples/tau_retail.md`.
2. Implement ROADMAP item 2 (Wilcoxon companion test).
3. After item 1 validates: PyPI release (item 9).

## Non-negotiable constraints

- **Core stays pure-Python, zero dependencies.** No numpy/scipy/pytest anywhere in `reliagate/metrics.py`, `compare.py`, `planning.py`, `gate.py`, `model.py`. Tests must run green with `python3 -m unittest` on a box that cannot pip install (PEP 668). Heavy numerics go behind lazy optional imports in `reliagate/adapters/` only.
- **Abstentions are never failures.** `success=None` counts in `n`, never in `c`; every report surfaces the treatment. Do not "fix" this by dropping them silently.
- **Exit codes are a contract.** 0 = pass, 2 = statistically significant regression, 3 = infra/usage error. Never let argparse or an unhandled exception make exit 2 mean anything else; never exit 0 on infra failure.
- **Live endpoint policy (owner rule).** All live agent runs use the Z.ai Coding Plan endpoint ONLY: base URL `https://api.z.ai/api/coding/paas/v4`, model `glm-5.3-flash` (pinned -- do not add or default to any other model anywhere in code/docs/examples), key from env var `RELIAGATE_ZAI_KEY`. The live path is opt-in behind that env var; core stats and tests are offline and key-free. On a billing or endpoint error: STOP and report -- never fall back to another provider or a pay-per-call endpoint.
- **Honest status separation.** "mechanism implemented + offline-tested" is not "benchmarks reproduced". Don't blur that line in the README until ROADMAP item 1 lands.
- **No AGENTS.md files** (blocked by harness) -- this CONTINUE.md is the handoff artifact; keep it current.
- Conventional commits in English; git user Gjusev <yoa.yoa91@gmail.com>.

## How to verify your work

```bash
cd /root/reliagate
python3 -m unittest          # must stay green, currently 90 tests
python3 -m reliagate --version
```

After pushing: `gh api repos/Gjusev/reliagate/contents/README.md --jq .name` to verify the push landed.
