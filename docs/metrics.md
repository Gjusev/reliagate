# Metrics

Notation: a task `x` is run `n` times and succeeds `c` times. Abstained trials (`success: null`) count in `n` and never in `c`.

## Reliability of one version

| API | Definition | Verified against |
|---|---|---|
| `pass_k(run, k)` / `pass_pow_k` | **pass^k**: mean over tasks of `C(c,k) / C(n,k)`, the chance that `k` runs drawn without replacement *all* succeed | brute-force k-subset enumeration; τ-bench, arXiv:2406.12045 §3 |
| `pass_at_k(run, k)` | **pass@k** (unbiased): mean of `1 − C(n−c,k) / C(n,k)`, the chance that *at least one* of `k` runs succeeds | brute-force enumeration plus the Codex Appendix A unbiasedness identity, checked in exact rational arithmetic (arXiv:2107.03374) |
| `consistency(run)` | each task is **consistent** (all `n` succeed), **sometimes**, or **never**; a task with an abstained trial cannot be consistent | arXiv:2604.17849 §2.2 |
| `wilson_ci`, `pass_k(..., ci="wilson_indicator")` | Wilson score interval; on the Pass^n task indicator it is a binomial proportion over tasks | published anchor 0/10 → [0, 0.2775]; non-degenerate at p̂ = 0 or 1 (arXiv:2605.08261 §4.2) |
| `pass_k(..., ci="bootstrap_tasks")` | percentile bootstrap that resamples **tasks**, never the trials inside a task | arXiv:2605.08261 §4.3 (resampling rollouts instead reaches only 17% coverage) |
| `hierarchical_bootstrap` | resamples scenarios, then tasks; per-task aggregates held fixed | deterministic seeds; arXiv:2605.08261 §4.3 |

**pass^k vs pass@k.** pass@k answers "can the agent ever do this?". pass^k answers "can I rely on it?". As `k` grows they move apart. In the [first study](results/first-study.md) the gap reaches +0.20 at k=4. A regression that only makes tasks *flaky* is invisible to pass@k, as the [example pair](getting-started.md#walkthrough-the-example-pair) shows: pass@4 = 1.00 on both versions, pass^4 0.85 → 0.65.

## Comparing two versions (paired, same tasks)

| API | Definition | Verified against |
|---|---|---|
| `compare(base, new)` | McNemar on the per-task indicator `z_x = 1[c_x == n]`. `b` = tasks that became consistent, `c` = tasks that stopped being consistent. Exact test when fewer than 25 pairs are discordant, mid-p otherwise | hand-computed exact and mid-p values, including the b=5, c=0 case where they disagree (Fagerland et al. 2013) |
| `.wilcoxon` | Wilcoxon signed-rank on `d_x = c_new − c_base`. Catches partial shifts that the binary indicator discards. Exact for ≤ 20 non-zero pairs, tie-corrected normal approximation above | hand-computed small cases; brute-force sign enumeration |
| `.newcombe` | hybrid-score interval for the difference of the two paired proportions (Newcombe 1998, method 10). McNemar alone gives no interval | interval contains the point estimate on randomized tables |
| `compare(..., task_categories=, correction="holm" \| "bh")` | per-category McNemar with Holm (FWER, for gates) or Benjamini-Hochberg (FDR, for dashboards) correction | hand-computed 4-p family |

## Planning and early stopping

| API | Definition |
|---|---|
| `required_runs(effect, alpha, power, p_base)` | smallest number of paired tasks that detects `effect` at the target power, by seeded simulation. The assumptions are listed in the result |
| `plan(pilot, effect)` | same, with `p_base` estimated from a pilot run |
| `sprt_stop(base, candidate_so_far, p1)` | Wald SPRT over the candidate's per-task indicators. Hitting the cap returns `continue`, which is **inconclusive and never a pass** |
| `consistency_settled(failures, done, planned)` | per-task early stop: one failure already rules out "consistent" |

## References

- Yao et al., τ-bench, arXiv:2406.12045 (pass^k, §3)
- Chen et al., *Evaluating Large Language Models Trained on Code*, arXiv:2107.03374 (unbiased pass@k, §2.1 and App. A)
- *On the Reliability of Computer Use Agents*, arXiv:2604.17849 (§2.2 consistency classes, McNemar on z_x)
- *Computer Use at the Edge of the Statistical Precipice*, arXiv:2605.08261 (§4.2 Wilson, §4.3 task-level bootstrap)
- Fagerland, Lydersen & Laake 2013, *Stat Med* 32:3086 (exact vs mid-p McNemar)
- Newcombe 1998, *Stat Med* 17:2635 (paired-difference intervals)
- tau2-bench issue #497 (the infra-error convention ambiguity reliagate's abstention rule closes)

reliagate implements these papers independently and is not affiliated with their authors.
