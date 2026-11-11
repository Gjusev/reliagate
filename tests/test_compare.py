"""McNemar exact vs mid-p: hand-computable cases and agreement rules."""

import fractions
import math
import unittest

from reliagate.compare import (
    compare,
    mcnemar_exact,
    mcnemar_midp,
    select_test,
)
from reliagate.model import EvalRun

from .test_estimators_property import make_run


class TestExact(unittest.TestCase):
    def test_hand_computed_values(self):
        # b=5, c=0: p = 2 * (1/32) = 0.0625 (exact, conservative)
        self.assertAlmostEqual(mcnemar_exact(5, 0), 0.0625)
        # b=6, c=0: p = 2/64 = 0.03125
        self.assertAlmostEqual(mcnemar_exact(6, 0), 0.03125)
        # b=5, c=1, m=6: 2*(P(0)+P(1)) = 2*(7/64) = 0.21875
        self.assertAlmostEqual(mcnemar_exact(5, 1), fractions.Fraction(14, 64))

    def test_exact_is_exact_rational(self):
        # 2 * sum_{i<=k} C(m,i) / 2^m computed as pure integer tail.
        p = mcnemar_exact(3, 0)  # 2 * (1/8) = 0.25
        self.assertEqual(fractions.Fraction(p).limit_denominator(10**6),
                         fractions.Fraction(1, 4))

    def test_no_discordant_pairs(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_midp(0, 0), 1.0)

    def test_symmetric(self):
        self.assertEqual(mcnemar_exact(4, 1), mcnemar_exact(1, 4))


class TestMidP(unittest.TestCase):
    def test_midp_equals_exact_minus_half_observed_cell(self):
        for b, c in ((5, 0), (6, 1), (3, 2), (10, 2), (30, 5)):
            m = b + c
            k = min(b, c)
            expected = mcnemar_exact(b, c) - math.comb(m, k) / 2**m
            self.assertAlmostEqual(mcnemar_midp(b, c), expected, msg=f"{b},{c}")

    def test_hand_computed_agreement_disagreement_case(self):
        # THE canonical disagreement case: b=5, c=0.
        # exact: 2/32 = 0.0625  -> NOT significant at 0.05 (conservative)
        # mid-p: 1/32 = 0.03125 -> significant
        exact = mcnemar_exact(5, 0)
        midp = mcnemar_midp(5, 0)
        self.assertAlmostEqual(exact, 0.0625)
        self.assertAlmostEqual(midp, 0.03125)
        self.assertGreaterEqual(exact, 0.05)
        self.assertLess(midp, 0.05)

    def test_midp_always_leq_exact(self):
        for b in range(0, 12):
            for c in range(0, 12):
                self.assertLessEqual(mcnemar_midp(b, c), mcnemar_exact(b, c) + 1e-15)

    def test_midp_never_negative(self):
        for b in range(0, 12):
            for c in range(0, 12):
                self.assertGreaterEqual(mcnemar_midp(b, c), 0.0)

    def test_both_are_two_sided(self):
        self.assertAlmostEqual(mcnemar_exact(7, 2), mcnemar_exact(2, 7))
        self.assertAlmostEqual(mcnemar_midp(7, 2), mcnemar_midp(2, 7))


class TestSelection(unittest.TestCase):
    def test_auto_uses_exact_below_25(self):
        self.assertEqual(select_test(24), "mcnemar_exact")
        self.assertEqual(select_test(3), "mcnemar_exact")

    def test_auto_switches_to_midp_at_25(self):
        self.assertEqual(select_test(25), "mcnemar_midp")
        self.assertEqual(select_test(100), "mcnemar_midp")


class TestCompare(unittest.TestCase):
    def test_counts_and_class_changes(self):
        base = make_run({
            "t1": [True, True],    # consistent
            "t2": [True, False],   # sometimes
            "t3": [False, False],  # never
        })
        new = make_run({
            "t1": [False, False],  # consistent -> never (a LOSS)
            "t2": [True, True],    # sometimes -> consistent (a GAIN)
            "t3": [True, True],    # never -> consistent (a GAIN)
        })
        result = compare(base, new)
        self.assertEqual(result.b, 2)
        self.assertEqual(result.c, 1)
        self.assertEqual(result.b_minus_c, 1)
        self.assertEqual(result.discordant, 3)
        self.assertEqual(result.test_used, "mcnemar_exact")
        self.assertAlmostEqual(result.p_value, 1.0)  # b=2,c=1 -> 2*(4/8)=1.0
        changes = {(ch["task_id"], ch["base"], ch["new"]) for ch in result.class_changes}
        self.assertEqual(changes, {
            ("t1", "consistent", "never"),
            ("t2", "sometimes", "consistent"),
            ("t3", "never", "consistent"),
        })

    def test_significant_regression_detected(self):
        base = make_run({f"t{i}": [True, True] for i in range(6)})
        # all 6 tasks lose consistency -> b=0, c=6
        new = make_run({
            f"t{i}": [True, False] for i in range(6)
        })
        result = compare(base, new)
        self.assertEqual(result.c, 6)
        self.assertEqual(result.b, 0)
        self.assertLess(result.p_value, 0.05)
        self.assertAlmostEqual(result.p_value, 2 / 64)  # exact: 2*(1/64)

    def test_mismatched_task_sets_raise(self):
        base = make_run({"a": [True], "b": [True]})
        new = make_run({"a": [True], "z": [True]})
        with self.assertRaises(ValueError):
            compare(base, new)

    def test_identical_runs_are_null(self):
        runs = make_run({f"t{i}": [True, True] for i in range(4)})
        result = compare(runs, EvalRun(name="copy", runs=runs.runs))
        self.assertEqual(result.b, 0)
        self.assertEqual(result.c, 0)
        self.assertEqual(result.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
