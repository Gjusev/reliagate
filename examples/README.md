# Example pair

`base.jsonl` (v1) and `new.jsonl` (v2): the same 40 tasks, 4 runs each, before and after a change.
In v2, nine tasks that passed every run now fail some of the time, and one flaky task became consistent.

```console
$ reliagate gate --base examples/base.jsonl --new examples/new.jsonl
gate decision: REGRESSION (exit 2)
  consistently solved: base 34/40 -> new 26/40
  gained (b)=1  lost (c)=9  b-c=-8  test=mcnemar_exact  p=0.02148
```

pass@4 is 1.00 on both files; pass^4 drops from 0.85 to 0.65. The files are synthetic (generated
with a fixed seed to illustrate a flakiness regression). For results on real agent trajectories,
see [docs/results/first-study.md](../docs/results/first-study.md).
