# CONTINUE.md -- handoff for coding agents

## Where this stopped

v0.1.0 + first real study + deferred statistics are committed. The library is complete, offline-tested, and validated on real data:

- `reliagate/model.py` -- Run/EvalRun, abstention accounting (None = abstained, counted in n, never a failure)
- `reliagate/metrics.py` -- pass^k, unbiased pass@k, consistency classes, Wilson CI, task-level bootstrap, rule-of-three
- `reliagate/compare.py` -- McNemar exact (< 25 discordant) / mid-p (>= 25), class-change table, PLUS wilcoxon + newcombe companions and per-category Holm/BH correction
- `reliagate/stats.py` -- Wilcoxon signed-rank, Holm/BH, hierarchical bootstrap, Newcombe paired interval (ROADMAP items 2, 3, 4, 7)
- `reliagate/planning.py` -- `required_runs`/`plan` (seeded simulation), `sprt_stop`, `consistency_settled`
- `reliagate/gate.py` + `reliagate/cli.py` -- gate CLI with exit codes 0/2/3
- `reliagate/adapters/` -- JSONL (stdlib), nitelos (lazy mapping), runner with concurrency + cost budget and the opt-in live Z.ai path
- `scripts/first_study.py` + `docs/results/first-study.md` -- the first real study (gate-calibrated tau-bench retail), deterministic and rerunnable, no LLM calls
- `tests/` -- 141 tests, all green with `python3 -m unittest`

ROADMAP items 1-4 and 7 are done. Remaining: Inspect/Langfuse adapters (5, 6), alpha spending (8), PyPI release (9).

## Next steps (in order)

1. Consider a second dataset (e.g. tau-bench airline) -- the first study's policy comparisons are all null/underpowered (~0.03-0.05 effects need 700-2200 paired tasks).
2. Implement ROADMAP item 5 or 6 (adapter stubs).
3. PyPI release (item 9).

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
