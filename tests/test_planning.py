"""Planning (simulation-based) and SPRT stopping."""

import math
import random
import unittest

from reliagate.model import EvalRun, Run
from reliagate.planning import (
    SprtResult,
    consistency_settled,
    plan,
    required_runs,
    sprt_stop,
)

from .test_estimators_property import make_run


class TestRequiredRuns(unittest.TestCase):
    def test_grows_as_effect_shrinks(self):
        big = required_runs(effect=0.4, p_base=0.5, seed=1)
        small = required_runs(effect=0.1, p_base=0.5, seed=1)
        self.assertGreater(small.n_tasks, big.n_tasks)

    def test_detects_real_effect_at_n_tasks(self):
        # with p_base=0.2, effect=0.6 (p_new=0.8), a small study must
        # reach target power -- sanity check the simulation machinery
        result = required_runs(effect=0.6, p_base=0.2, power=0.8, seed=2)
        self.assertLessEqual(result.n_tasks, 60)
        self.assertGreaterEqual(result.achieved_power, 0.8)

    def test_deterministic_seed(self):
        r1 = required_runs(effect=0.3, seed=5)
        r2 = required_runs(effect=0.3, seed=5)
        self.assertEqual(r1.n_tasks, r2.n_tasks)
        self.assertEqual(r1.achieved_power, r2.achieved_power)

    def test_validates_ranges(self):
        with self.assertRaises(ValueError):
            required_runs(effect=0.0)
        with self.assertRaises(ValueError):
            required_runs(effect=1.2)
        with self.assertRaises(ValueError):
            required_runs(effect=0.2, p_base=0.9)  # p_base+effect > 1

    def test_documents_assumptions(self):
        result = required_runs(effect=0.3, p_base=0.4)
        self.assertTrue(result.assumptions)
        joined = " ".join(result.assumptions)
        self.assertIn("Bernoulli", joined)
        self.assertIn("independent", joined)

    def test_max_tasks_cap_reports_honestly(self):
        result = required_runs(effect=0.02, p_base=0.5, power=0.99,
                               max_tasks=16)
        self.assertIn("not reached", " ".join(result.assumptions))


class TestPlanFromPilot(unittest.TestCase):
    def test_pilot_estimates_p_base(self):
        # pilot: 2 of 4 tasks consistently solved -> p_base = 0.5
        pilot = make_run({
            "a": [True, True], "b": [True, True],
            "c": [True, False], "d": [False, False],
        })
        result = plan(pilot, effect=0.5, seed=3)
        self.assertIn("p_base=0.5 estimated", result.assumptions[0])


class TestConsistencySettled(unittest.TestCase):
    def test_one_failure_settles_to_zero(self):
        self.assertTrue(consistency_settled(failures_so_far=1,
                                            trials_done=1, n_planned=5))
        self.assertTrue(consistency_settled(failures_so_far=2,
                                            trials_done=4, n_planned=5))

    def test_all_success_through_planned_settles_to_one(self):
        self.assertTrue(consistency_settled(0, 5, 5))
        self.assertFalse(consistency_settled(0, 3, 5))

    def test_validates(self):
        with self.assertRaises(ValueError):
            consistency_settled(0, 0, 0)


class TestSprt(unittest.TestCase):
    def _base(self, p=0.5):
        rng = random.Random(11)
        runs = []
        for t in range(20):
            for a in range(2):
                runs.append(Run(task_id=f"t{t}", attempt=a,
                                success=rng.random() < p))
        return EvalRun(name="base", runs=runs)

    def test_accepts_clearly_better_candidate(self):
        base = self._base()
        # candidate: everything consistently solved -> LLR climbs fast
        cand = make_run({f"t{i}": [True, True] for i in range(20)})
        result = sprt_stop(base, cand, p1=0.9)
        self.assertEqual(result.decision, "accept")
        self.assertGreaterEqual(result.llr, result.upper)

    def test_rejects_clearly_worse_candidate(self):
        base = self._base()
        cand = make_run({f"t{i}": [False, False] for i in range(20)})
        result = sprt_stop(base, cand, p1=0.9)
        self.assertEqual(result.decision, "reject")
        self.assertLessEqual(result.llr, result.lower)

    def test_continue_when_ambiguous(self):
        base = make_run({f"t{i}": [True, True] for i in range(5)}
                        | {f"s{i}": [True, False] for i in range(5)})
        runs = [Run(task_id=f"t{i}", attempt=0, success=True) if i % 2 == 0
                else Run(task_id=f"t{i}", attempt=0, success=False)
                for i in range(4)]
        cand = EvalRun(name="cand", runs=runs)
        result = sprt_stop(base, cand, p1=0.9)
        self.assertEqual(result.decision, "continue")

    def test_cap_is_inconclusive_not_pass(self):
        base = make_run({f"t{i}": [True, True] for i in range(5)}
                        | {f"s{i}": [True, False] for i in range(5)})
        # one gain (+2.89) and one loss (-1.61): LLR stays between the
        # Wald boundaries, so the cap must produce an honest 'continue'
        cand = make_run({"t0": [True, True], "t1": [True, False]})
        result = sprt_stop(base, cand, p1=0.9, max_tasks=2)
        self.assertEqual(result.decision, "continue")
        self.assertEqual(result.n_tasks, 2)
        self.assertTrue(any("INCONCLUSIVE" in n for n in result.notes))

    def test_thresholds_match_wald(self):
        base = self._base()
        cand = make_run({f"t0": [True, True]})
        result = sprt_stop(base, cand, p1=0.9, alpha=0.05, beta=0.10)
        self.assertAlmostEqual(result.upper, math.log(0.9 / 0.05))
        self.assertAlmostEqual(result.lower, math.log(0.10 / 0.95))

    def test_rejects_degenerate_p0(self):
        all_fail = make_run({f"t{i}": [False] for i in range(5)})
        with self.assertRaises(ValueError):
            sprt_stop(all_fail, make_run({"a": [True]}), p1=0.9)


if __name__ == "__main__":
    unittest.main()
