# Changelog

## 0.2.0 (2026-11-19)

Gate behavior changes. Check your CI if you relied on the old exit codes.

- **New exit code 4, inconclusive.** A drop beyond `--max-regression` that is not statistically significant used to pass (exit 0). It now returns 4: "not significant" is not "safe".
- **New `--max-abstention` (default 0.05).** If either version has more abstained or infra-errored trials than this share, the gate returns 3 instead of a verdict, so infrastructure failures can't show up as an agent regression.
- **Same number of trials per task required.** `compare` and the gate reject pairs where a task was run a different number of times in each version (exit 3). Before, unequal `k` could produce a false significant regression.
- **Fixed power simulation.** `required_runs`/`plan` drew gains and losses independently, so one task could count as both. Each task now draws one (base, new) pair. Planned task counts shift slightly; the first study's conclusions are unchanged.
- **CLI:** invalid `--alpha` / `--max-regression` exits 3 instead of crashing with 1.
- **Docs:** new README, `docs/` guides, `examples/` pair and launch film. The README quickstart runs as a test.

## 0.1.0 (2026-11-16)

First release: pass^k, unbiased pass@k, consistency classes, Wilson and bootstrap intervals, paired McNemar with Wilcoxon and Newcombe companions, Holm/BH corrections, planning, SPRT, and the CI gate.
