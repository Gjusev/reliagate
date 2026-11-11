"""Wilson interval, consistency classes, abstention handling, bootstrap."""

import math
import unittest

from reliagate.model import EvalRun, Run, abstention_report, consistency_class
from reliagate.metrics import (
    bootstrap_pass_pow_k,
    pass_k,
    wilson_ci,
    wilson_indicator,
)

from .test_estimators_property import make_run


class TestWilson(unittest.TestCase):
    def test_known_value_zero_out_of_ten(self):
        # Published anchor: Wilson 95% interval for 0/10 successes is
        # [0, 0.2775...] (upper bound ~3/10 minus small correction).
        lo, hi = wilson_ci(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.27754, places=4)

    def test_never_degenerate_at_boundaries_unlike_wald(self):
        # Wald collapses to zero width at p_hat=0/1; Wilson must not.
        for succ, total in ((0, 10), (10, 10), (0, 3), (3, 3)):
            lo, hi = wilson_ci(succ, total)
            self.assertLess(lo, hi, msg=f"{succ}/{total}")
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_contains_point_estimate_interior(self):
        for succ in range(1, 10):
            lo, hi = wilson_ci(succ, 10)
            p = succ / 10
            self.assertLessEqual(lo, p + 1e-12)
            self.assertGreaterEqual(hi, p - 1e-12)

    def test_narrows_with_more_data(self):
        lo3, hi3 = wilson_ci(1, 3)
        lo30, hi30 = wilson_ci(10, 30)
        self.assertLess(hi30 - lo30, hi3 - lo3)

    def test_validates_input(self):
        with self.assertRaises(ValueError):
            wilson_ci(5, 0)
        with self.assertRaises(ValueError):
            wilson_ci(11, 10)
        with self.assertRaises(ValueError):
            wilson_ci(1, 10, alpha=1.5)


class TestConsistencyClasses(unittest.TestCase):
    def test_three_way_partition(self):
        self.assertEqual(consistency_class(3, 3), "consistent")
        self.assertEqual(consistency_class(2, 3), "sometimes")
        self.assertEqual(consistency_class(0, 3), "never")

    def test_abstention_blocks_consistent(self):
        # a task with an abstained trial is not 'consistently' solved:
        # consistent requires every counted trial to be a success.
        self.assertEqual(consistency_class(2, 3), "sometimes")


class TestAbstention(unittest.TestCase):
    def test_abstained_in_denominator_not_in_successes(self):
        run = make_run({"a": [True, None, True]})
        self.assertEqual(pass_k(run, 1).value, 2 / 3)  # 2 successes / 3 trials

    def test_drop_abstained_option(self):
        run = make_run({"a": [True, None, True]})
        self.assertEqual(pass_k(run, 1, drop_abstained=True).value, 1.0)

    def test_report_states_treatment(self):
        run = make_run({"a": [True, None], "b": [False, None, None]})
        report = abstention_report(run)
        self.assertEqual(report["trials_total"], 5)
        self.assertEqual(report["trials_abstained"], 3)
        self.assertEqual(report["tasks_with_abstentions"], 2)
        self.assertIn("never", report["treatment"])

    def test_abstained_never_fails_a_task_into_never(self):
        run = make_run({"a": [True, None]})
        rows = pass_k(run, 1)
        self.assertEqual(rows.value, 0.5)


class TestBootstrap(unittest.TestCase):
    def test_deterministic_with_seed(self):
        run = make_run({
            "a": [True] * 5, "b": [True] * 5, "c": [True, False, True, True, True],
            "d": [False] * 5,
        })
        e1 = bootstrap_pass_pow_k(run, 2, seed=7)
        e2 = bootstrap_pass_pow_k(run, 2, seed=7)
        self.assertEqual(e1.ci, e2.ci)
        self.assertEqual(e1.value, e2.value)

    def test_ci_contains_point_for_symmetric_data(self):
        run = make_run({
            "a": [True, True], "b": [True, True], "c": [False, False],
            "d": [False, False],
        })
        est = bootstrap_pass_pow_k(run, 1, n_boot=500, seed=1)
        self.assertLessEqual(est.ci[0], est.value)
        self.assertLessEqual(est.value, est.ci[1])

    def test_ci_wider_for_heterogeneous_tasks(self):
        homog = make_run({
            f"t{i}": [True, True] for i in range(10)
        } | {"x": [False, False]})
        est = bootstrap_pass_pow_k(homog, 1, n_boot=300, seed=3)
        self.assertGreater(est.ci[1] - est.ci[0], 0)

    def test_wilson_indicator_method(self):
        run = make_run({
            "a": [True, True], "b": [True, True], "c": [True, False],
        })
        est = wilson_indicator(run)
        self.assertAlmostEqual(est.value, 2 / 3)
        self.assertIsNotNone(est.ci)
        lo, hi = est.ci
        self.assertLessEqual(lo, est.value + 1e-12)
        self.assertGreaterEqual(hi, est.value - 1e-12)


if __name__ == "__main__":
    unittest.main()
