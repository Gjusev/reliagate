# Design and limitations

## Decisions

- **The task is the unit of inference.** Trials within a task are not independent evidence. Intervals resample tasks, and comparisons pair tasks across versions.
- **Abstentions are never failures.** `success: null` stays in the denominator `n` and is excluded from successes `c`, the conservative direction. Every report states this treatment, so the convention can't change silently (the ambiguity tau2-bench hit in issue #497).
- **"Consistent" needs decisive evidence.** A task with an abstained trial can't be consistent, because consistency claims success on all `n` runs.
- **Exact integer combinatorics.** Estimators use `math.comb`, matching the papers' closed forms without float accumulation.
- **Zero dependencies.** The core is stdlib only (`math.comb`, `math.erf`, seeded `random.Random`). The test suite runs with plain `python -m unittest` on Linux, Windows and macOS, Python 3.10 to 3.13.
- **Deterministic.** Every bootstrap and simulation takes a seed and repeats exactly.

## Running agents (optional)

reliagate's statistics never touch the network. To collect trials, pass any function `task_id -> bool | None` to `reliagate.adapters.runner.run_agent`, which handles repetition, a thread pool and a cost budget. One ready-made agent function ships as an example: `make_zai_agent_fn`, an OpenAI-compatible client for the Z.ai endpoint. It is opt-in behind the `RELIAGATE_ZAI_KEY` environment variable and stops on billing or endpoint errors instead of falling back to another provider.

## Limitations

- **Pairs, not leaderboards.** `compare` tests two versions of one agent on one task set. It does not rank many agents.
- **Binary outcomes.** Trials are pass, fail or abstain. Partial-credit scores are not modeled yet.
- **Small samples are small.** The exact McNemar test is conservative and mid-p is closer to nominal, but neither is magic. Below ~10 tasks the gate's "not significant" often means "underpowered". Size the study with `plan()` first. Bootstrap intervals with fewer than 20 tasks are wide by construction.
- **Planning assumptions are simple.** `required_runs` simulates independent Bernoulli indicators. Real paired runs share task difficulty, which usually makes real power *better* than simulated. This has not yet been verified against real data.

## Non-goals (for now)

- Multi-model leaderboards and ranking.
- Partial-credit scores in the core data model.
- Web dashboards. The gate is a CLI; visualization belongs to the caller.
