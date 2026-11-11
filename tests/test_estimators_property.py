"""Property tests: estimators vs brute-force enumeration.

Every identity here is checked against itertools brute force over small
n, not against a re-implementation of the same formula.
"""

import fractions
import itertools
import math
import unittest

from reliagate.model import EvalRun, Run
from reliagate.metrics import (
    consistency,
    pass_at_k,
    pass_pow_k,
    task_pass_at_k,
    task_pass_pow_k,
)


def make_run(per_task_outcomes):
    """per_task_outcomes: {task_id: [True/False/None, ...]}"""
    runs = []
    for tid, outcomes in per_task_outcomes.items():
        for attempt, s in enumerate(outcomes):
            runs.append(Run(task_id=tid, attempt=attempt, success=s))
    return EvalRun(name="t", runs=runs)


def brute_force_pass_pow_k(outcomes, k):
    """Mean over ALL k-subsets of trials of indicator(all success)."""
    n = len(outcomes)
    subsets = list(itertools.combinations(range(n), k))
    hits = sum(1 for subset in subsets if all(outcomes[i] for i in subset))
    return hits / len(subsets)


def brute_force_pass_at_k(outcomes, k):
    """Mean over ALL k-subsets of indicator(at least one success)."""
    n = len(outcomes)
    subsets = list(itertools.combinations(range(n), k))
    hits = sum(1 for subset in subsets if any(outcomes[i] for i in subset))
    return hits / len(subsets)


class TestPassPowKBruteForce(unittest.TestCase):
    def test_matches_subset_enumeration(self):
        # C(c,k)/C(n,k) must equal the mean of the all-success indicator
        # over every k-subset of trials (tau-bench arXiv:2406.12045 s.3).
        for n in range(1, 8):
            for c in range(0, n + 1):
                outcomes = [True] * c + [False] * (n - c)
                for k in range(1, n + 1):
                    expected = brute_force_pass_pow_k(outcomes, k)
                    got = task_pass_pow_k(c, n, k)
                    self.assertAlmostEqual(
                        expected, got, msg=f"c={c} n={n} k={k}"
                    )

    def test_run_level_matches_brute_force(self):
        run = make_run(
            {"a": [True, True, False], "b": [True, False, False], "c": [True] * 3}
        )
        for k in (1, 2, 3):
            expected = (
                brute_force_pass_pow_k([True, True, False], k)
                + brute_force_pass_pow_k([True, False, False], k)
                + brute_force_pass_pow_k([True, True, True], k)
            ) / 3
            self.assertAlmostEqual(pass_pow_k(run, k), expected)


class TestPassAtKBruteForce(unittest.TestCase):
    def test_matches_subset_enumeration(self):
        # 1 - C(n-c,k)/C(n,k) equals the mean any-success indicator over
        # k-subsets (Codex arXiv:2107.03374 s.2.1 Eq. 1).
        for n in range(1, 8):
            for c in range(0, n + 1):
                outcomes = [True] * c + [False] * (n - c)
                for k in range(1, n + 1):
                    expected = brute_force_pass_at_k(outcomes, k)
                    got = task_pass_at_k(c, n, k)
                    self.assertAlmostEqual(
                        expected, got, msg=f"c={c} n={n} k={k}"
                    )

    def test_boundary(self):
        # n-c < k => every k-subset contains a success => exactly 1.0
        self.assertEqual(task_pass_at_k(3, 5, 3), 1.0)
        self.assertEqual(task_pass_at_k(0, 5, 1), 0.0)


class TestCodexUnbiasedness(unittest.TestCase):
    def test_expectation_is_exactly_1_minus_1_minus_p_pow_k(self):
        # Codex Appendix A identity, verified EXACTLY with rational
        # arithmetic: E_c[1 - C(n-c,k)/C(n,k)] = 1 - (1-p)^k for
        # c ~ Binomial(n, p).  This is the property the naive
        # 1-(1-p_hat)^k estimator provably lacks.
        for n in range(2, 9):
            for k in range(1, n + 1):
                for p in (fractions.Fraction(1, 5), fractions.Fraction(1, 2),
                          fractions.Fraction(9, 10)):
                    expectation = fractions.Fraction(0)
                    for c in range(0, n + 1):
                        weight = (
                            math.comb(n, c)
                            * p**c
                            * (1 - p) ** (n - c)
                        )
                        estimator = fractions.Fraction(
                            math.comb(n, k) - math.comb(n - c, k),
                            math.comb(n, k),
                        )
                        expectation += weight * estimator
                    target = 1 - (1 - p) ** k
                    self.assertEqual(
                        expectation, target, msg=f"n={n} k={k} p={p}"
                    )

    def test_tau_bench_identity_pass1(self):
        # tau-bench s.3: pass^1 == pass@1 == E[c/n].
        for n in range(1, 7):
            for c in range(0, n + 1):
                self.assertAlmostEqual(
                    task_pass_pow_k(c, n, 1), task_pass_at_k(c, n, 1), places=15
                )
                self.assertAlmostEqual(task_pass_pow_k(c, n, 1), c / n, places=15)

    def test_pass_pow_n_is_consistency_fraction(self):
        # arXiv:2604.17849 s.2.2: Pass^n == fraction of tasks with c_x == n.
        run = make_run(
            {"a": [True, True, True], "b": [True, False, True],
             "c": [False, False, False]}
        )
        self.assertEqual(pass_pow_k(run, 3), 1 / 3)
        self.assertEqual(consistency(run)["pass_pow_n"], 1 / 3)
        self.assertEqual(consistency(run)["counts"],
                         {"consistent": 1, "sometimes": 1, "never": 1,
                          "no_trials": 0})


class TestValidation(unittest.TestCase):
    def test_rejects_k_greater_than_n(self):
        with self.assertRaises(ValueError):
            task_pass_pow_k(1, 2, 3)

    def test_run_level_k_above_task_trials_raises(self):
        run = make_run({"a": [True, True], "b": [True, False]})
        with self.assertRaises(ValueError):
            pass_pow_k(run, 3)

    def test_duplicate_trial_rejected(self):
        with self.assertRaises(ValueError):
            EvalRun(name="t", runs=[
                Run("a", 0, True), Run("a", 0, False),
            ])


if __name__ == "__main__":
    unittest.main()
