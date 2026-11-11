# NOTES.md -- session notes

## Done

- Core library built per the 2A research API (read from `/root/work/opensource-plan/2A-reliability-sota.md`):
  - Data model: `Run(task_id, attempt, success: bool | None, version, metadata)`; `EvalRun` rejects duplicate (task_id, attempt) pairs.
  - pass^k and unbiased pass@k via exact `math.comb`; consistency classes; Wilson CI (z via bisection on `math.erf`); task-level percentile bootstrap with fixed seeds; rule-of-three companion bound.
  - Paired McNemar on `z_x = 1[c_x == n]` with the auto rule: exact when discordant < 25, mid-p otherwise (Fagerland et al. 2013 rationale in docstrings).
  - Simulation-based `required_runs(effect, alpha, power, p_base)` + pilot-based `plan()`; SPRT stopping with Wald boundaries; `consistency_settled` for per-task early stop.
  - Gate CLI `reliagate gate` with exit codes 0 / 2 / 3 and usage errors forced to 3 (custom argparse class).
  - Adapters: JSONL round-trip with strict validation; nitelos verdict mapping (lazy, optional); generic `run_agent` with thread-pool concurrency, cost budget (estimated-cost gating), abstention-on-skip.
  - Live path per owner rule: Z.ai Coding Plan `https://api.z.ai/api/coding/paas/v4`, model glm-5.3-flash pinned (code rejects any other model), key via `RELIAGATE_ZAI_KEY`, opt-in only, no provider fallback on billing errors.
- 90 tests green via `python3 -m unittest`; zero dependencies.
- README / ROADMAP / CONTINUE.md / LICENSE (MIT, Youssef Ouhaghi 2026) / pyproject.toml written and pushed to github.com/Gjusev/reliagate with topics.

## What each metric is verified against (test provenance)

- pass^k, pass@k: brute-force k-subset enumeration over n<=7 (tests/test_estimators_property.py).
- Unbiasedness of pass@k: exact rational-arithmetic verification of the Codex Appendix A identity E[est] = 1-(1-p)^k for c~Binom(n,p) (same file).
- pass^1 == pass@1 == c/n: tau-bench §3 identity.
- Wilson 0/10 → [0, 0.2775]: published anchor; non-degeneracy at boundaries.
- McNemar exact/mid-p: hand-computed values (b=5,c=0 disagreement case: exact 0.0625 vs mid-p 0.03125), symmetry, mid-p <= exact, two-sidedness.
- Gate exit codes 0/2/3 incl. CLI subprocess runs and the "usage error is 3, not 2" rule.
- SPRT: Wald thresholds, accept/reject/continue, cap → honest INCONCLUSIVE.
- Planning: seeded determinism, monotonicity in effect size, documented-assumption checks.

## Left / deferred

- ROADMAP items 2-9 (Wilcoxon companion, Holm/BH correction, hierarchical bootstrap, Inspect AI + Langfuse adapters documented as stubs, Newcombe interval, group-sequential, PyPI).
- ROADMAP item 1 = first real-data run on Gjusev/gate-calibrated tau-bench retail trajectories -- NOT done; library status is "mechanism implemented + offline-tested", not "benchmarks reproduced".
- Inspect AI / Langfuse adapters are intentionally stubs in ROADMAP only; nothing faked.

## Doubts / caveats

- `run_agent` budget semantics: without `estimated_cost_per_call`, a batch can overshoot the budget by up to (batch size - 1) observed costs -- documented in the docstring, but a caller might expect hard clamping; hard clamping is impossible without knowing cost upfront.
- `required_runs` assumes independent base/new indicators; real paired runs share task difficulty, so simulated power is plausibly conservative, but this is unverified until ROADMAP item 1.
- Bootstrap percentile CI uses the simple index-rounding quantile; fine for B>=500, slightly coarse for tiny B.
- Python 3.10+ assumed (`X | None` syntax in signatures); tested only on 3.14 on this box.
- This box has no pytest; the suite is unittest-only by design, so no pytest plugins/features were used.
