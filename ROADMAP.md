# ROADMAP

Ordered by priority. Item 1 is deliberately first: everything else is unvalidated until the library touches real trajectories.

## 1. First real-data run (next)

Reuse the tau-bench **retail** trajectories already collected in `Gjusev/gate-calibrated` as the first real study:

1. Convert those trajectories into reliagate JSONL (`task_id`, `attempt`, `success`, `version`, `metadata`).
2. Recompute pass^1/pass^k and the consistency partition for the original agent; sanity-check pass^1 against the gate-calibrated study's own numbers.
3. Manufacture or find a genuine two-version comparison (e.g. temperature 0 vs default) and run `reliagate gate` on it; confirm the exact/mid-p selection behaves sensibly on real discordant-pair counts.
4. Write up `examples/tau_retail.md` with what matched, what didn't, and why.

Only after this can the README's "mechanism implemented" claims become "benchmarks validated".

## 2. Wilcoxon signed-rank companion test

`compare()` currently discards partial consistency shifts (1→2 of 3). Add the secondary test on `d_x = c_x_new - c_x_base` with average-rank ties handling (arXiv:2604.17849 §2.2), plus `delta_cx` effect size. Pure-python ranks are easy; the mid-p-style exact distribution for small samples needs care.

## 3. Multiple-comparison correction for task categories

`compare(..., family="categories")`: Holm step-down for confirmatory gates, BH for exploratory dashboards (2A research §3.5). Also the max-statistic bootstrap calibration when categories < ~20.

## 4. Hierarchical bootstrap

Tasks nesting in suites/apps/categories: resample within app (scenarios → axis levels → rollouts), apps fixed (arXiv:2605.08261 §4.3). Needed once real studies have category structure.

## 5. Adapter: Inspect AI (stub)

Read Inspect AI `eval_set` logs with `epochs=N` into EvalRun. Lazy import, optional dependency, tests skipped when inspect-ai is absent. Do NOT stub the metrics -- map their epoch reducers honestly or don't map them.

## 6. Adapter: Langfuse (stub)

Ingest Langfuse experiment scores for repeated runs per dataset item. Same rules as 5.

## 7. Newcombe/Tango interval for the paired difference

`compare()` reports b-c but no interval on the difference of proportions. Add the Newcombe hybrid-score interval (pure python; closed form).

## 8. Group-sequential alpha spending

O'Brien-Fleming checkpoints for CI that evaluates per PR (simpler to reason about than SPRT for that cadence). Complements `sprt_stop`.

## 9. PyPI release

Once item 1 is done and the README status line is honest about it.

## Non-goals (for now)

- Multi-model leaderboards / ranking (out of scope; single-model pair comparisons only).
- Partial-credit scores in the core data model ( Trials are pass/fail/abstain until a real study needs otherwise).
- Web dashboards. The gate is a CLI; visualization belongs to the caller.
