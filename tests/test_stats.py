"""Wilcoxon signed-rank, Holm/BH, hierarchical bootstrap, Newcombe: anchors
and property tests (pure stdlib, no reference-library cross-checks)."""

import itertools
import math
import random
import unittest

from reliagate.stats import (
    average_ranks,
    benjamini_hochberg_adjust,
    correct_pvalues,
    hierarchical_bootstrap,
    holm_adjust,
    newcombe_paired,
    wilcoxon_signed_rank,
)
from reliagate.metrics import wilson_ci


# ---------------------------------------------------------------------------
# average ranks


class TestAverageRanks(unittest.TestCase):
    def test_distinct_values(self):
        self.assertEqual(average_ranks([3.0, 1.0, 2.0]), [3.0, 1.0, 2.0])

    def test_ties_get_average_rank(self):
        # |d| = [0.1, 0.1, 0.2] -> ranks [1.5, 1.5, 3.0]
        self.assertEqual(average_ranks([0.1, 0.1, 0.2]), [1.5, 1.5, 3.0])

    def test_all_equal(self):
        self.assertEqual(average_ranks([5.0, 5.0, 5.0, 5.0]), [2.5] * 4)

    def test_ranks_sum_is_invariant(self):
        vals = [random.Random(7).random() if i % 3 else 0.5 for i in range(12)]
        self.assertAlmostEqual(sum(average_ranks(vals)),
                               sum(range(1, len(vals) + 1)))


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank


class TestWilcoxonExact(unittest.TestCase):
    def test_all_positive_n1(self):
        # d = [1.0]: W+ = 1, W- = 0, W = 0; exact two-sided p = 2/2 = 1.0
        r = wilcoxon_signed_rank([1.0])
        self.assertEqual(r.n_nonzero, 1)
        self.assertAlmostEqual(r.p_value, 1.0)
        self.assertEqual(r.method, "exact")

    def test_all_positive_n3_is_boundary(self):
        # d = [1, 2, 3]: ranks 1,2,3; W+ = 6, W- = 0, W = 0
        # exact p = P(W <= 0) = 2/8 = 0.25
        r = wilcoxon_signed_rank([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r.w, 0.0)
        self.assertAlmostEqual(r.p_value, 0.25)

    def test_single_negative_is_mirror(self):
        r = wilcoxon_signed_rank([-1.0, 2.0, 3.0])
        # W+ = 5, W- = 1, W = 1; p = P(W <= 1) counts subsets of {1,2,3}
        # with sum <= 1 ({}, {1}) plus complements ({2,3}, {3}) -> 4/8
        self.assertAlmostEqual(r.w, 1.0)
        self.assertAlmostEqual(r.p_value, 0.5)

    def test_hand_case_five_pairs_four_positive(self):
        # d = [1,2,3,4,-5]: W+ = 10, W- = 5, W = 5.  min(S+, 15-S+) <= 5
        # iff the positive subset sums to <= 5 or >= 10; there are 10
        # subsets with sum <= 5 and (by complement symmetry) 10 with
        # sum >= 10 -> p = 20/32.
        r = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, -5.0])
        self.assertAlmostEqual(r.p_value, 20 / 32)

    def test_extreme_imbalance_is_significant(self):
        # 16 positive values, one large negative: n=16, W- = 16, and the
        # exact p (2 * #{subsets of {1..16} with sum <= 16} / 2^16) is
        # well below 0.05.
        r = wilcoxon_signed_rank(
            [float(i) for i in range(1, 17)] + [-1000.0]
        )
        self.assertLess(r.p_value, 0.05)

    def test_brute_force_agreement_small_n(self):
        # The enumeration inside wilcoxon_signed_rank is the definition;
        # re-derive the two-sided p independently via itertools.
        rng = random.Random(11)
        for trial in range(20):
            d = [rng.uniform(-2, 2) for _ in range(rng.randint(1, 7))]
            r = wilcoxon_signed_rank(d)
            nz = [x for x in d if x != 0.0]
            if not nz:
                self.assertEqual(r.p_value, 1.0)
                continue
            ranks = average_ranks([abs(x) for x in nz])
            total = 1 << len(nz)
            hits = 0
            for signs in itertools.product((0, 1), repeat=len(nz)):
                sp = sum(rk for rk, s in zip(ranks, signs) if s == 1)
                sm = sum(rk for rk, s in zip(ranks, signs) if s == 0)
                if min(sp, sm) <= r.w:
                    hits += 1
            self.assertAlmostEqual(r.p_value, hits / total)

    def test_zeros_are_dropped(self):
        full = wilcoxon_signed_rank([0.0, 1.0, -2.0, 0.0])
        trimmed = wilcoxon_signed_rank([1.0, -2.0])
        self.assertEqual(full.n_nonzero, 2)
        self.assertAlmostEqual(full.p_value, trimmed.p_value)

    def test_all_zeros_is_degenerate(self):
        r = wilcoxon_signed_rank([0.0, 0.0, 0.0])
        self.assertEqual(r.method, "degenerate")
        self.assertEqual(r.p_value, 1.0)
        self.assertTrue(any("undefined" in n for n in r.notes))

    def test_symmetry_under_negation(self):
        rng = random.Random(3)
        d = [rng.uniform(-3, 3) for _ in range(6)]
        a = wilcoxon_signed_rank(d)
        b = wilcoxon_signed_rank([-x for x in d])
        self.assertAlmostEqual(a.p_value, b.p_value)
        self.assertAlmostEqual(a.w, b.w)

    def test_exact_rejects_large_n(self):
        with self.assertRaises(ValueError):
            wilcoxon_signed_rank([1.0] * 21, exact=True)


class TestWilcoxonNormal(unittest.TestCase):
    def test_normal_close_to_exact_mid_range(self):
        # n=12: exact and tie-corrected normal approx should be close
        d = [0.8, -0.3, 1.1, 0.2, -1.5, 2.0, 0.9, 0.1, -0.4, 1.3, 0.6, -0.9]
        exact = wilcoxon_signed_rank(d, exact=True)
        normal = wilcoxon_signed_rank(d, exact=False)
        self.assertAlmostEqual(exact.p_value, normal.p_value, delta=0.03)

    def test_continuity_correction_present(self):
        d = [float(i) for i in range(1, 26)] + [-1.5]
        r = wilcoxon_signed_rank(d, exact=False)
        self.assertEqual(r.method, "normal_approx")
        self.assertLess(r.p_value, 1e-3)  # extreme imbalance stays significant

    def test_tie_correction_applied(self):
        # Many ties -> note mentions the correction; result still valid
        d = [1.0] * 10 + [-1.0] * 10 + [0.3]
        r = wilcoxon_signed_rank(d, exact=False)
        self.assertTrue(any("tie correction" in n for n in r.notes))

    def test_alternatives(self):
        d = [2.0, 1.5, 1.0, 0.5, 1.2, 0.9]
        two = wilcoxon_signed_rank(d, "two-sided")
        greater = wilcoxon_signed_rank(d, "greater")
        less = wilcoxon_signed_rank(d, "less")
        self.assertAlmostEqual(greater.p_value, two.p_value / 2.0, delta=1e-9)
        self.assertLess(greater.p_value, 0.5)
        self.assertGreater(less.p_value, 0.5)
        with self.assertRaises(ValueError):
            wilcoxon_signed_rank(d, "bogus")


# ---------------------------------------------------------------------------
# multiple-comparison corrections


class TestHolm(unittest.TestCase):
    def test_hand_computed_example(self):
        # p = [.01, .02, .04, .05], m=4:
        # sorted: .01*4=.04 -> .02*3=.06 -> .04*2=.08 -> .05*1=.05 enforce
        # monotone -> [.04, .06, .08, .08]
        adj = holm_adjust([0.01, 0.02, 0.04, 0.05])
        self.assertAlmostEqual(adj[0], 0.04)
        self.assertAlmostEqual(adj[1], 0.06)
        self.assertAlmostEqual(adj[2], 0.08)
        self.assertAlmostEqual(adj[3], 0.08)

    def test_step_down_decision_matches_definition(self):
        # Holm rejects every p_(i) <= alpha/(m-i+1) in sorted order, stopping
        # at the first non-rejection; adjusted-p < alpha must agree exactly.
        rng = random.Random(5)
        for _ in range(30):
            ps = [rng.random() for _ in range(rng.randint(1, 8))]
            alpha = 0.05
            adj = holm_adjust(ps)
            order = sorted(range(len(ps)), key=lambda i: ps[i])
            expected_reject = []
            for rank, idx in enumerate(order, start=1):
                if ps[idx] <= alpha / (len(ps) - rank + 1):
                    expected_reject.append(idx)
                else:
                    break
            got_reject = [i for i in range(len(ps)) if adj[i] < alpha]
            self.assertEqual(sorted(got_reject), sorted(expected_reject),
                             msg=f"ps={ps}")

    def test_single_p_unchanged(self):
        self.assertAlmostEqual(holm_adjust([0.03]), [0.03])

    def test_never_decreases_or_exceeds_one(self):
        ps = [0.001, 0.5, 0.9, 0.02, 0.03, 0.04]
        adj = holm_adjust(ps)
        for a, p in zip(adj, ps):
            self.assertGreaterEqual(a, p)
            self.assertLessEqual(a, 1.0)

    def test_empty(self):
        self.assertEqual(holm_adjust([]), [])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            holm_adjust([1.5])


class TestBH(unittest.TestCase):
    def test_hand_computed_example(self):
        # p = [.01, .02, .04, .05], m=4:
        # from largest: .05*4/4=.05; .04*4/3=.0533->min(.05)=.05;
        # .02*4/2=.04; .01*4=.04 -> [.04, .04, .05, .05]
        adj = benjamini_hochberg_adjust([0.01, 0.02, 0.04, 0.05])
        self.assertAlmostEqual(adj[0], 0.04)
        self.assertAlmostEqual(adj[1], 0.04)
        self.assertAlmostEqual(adj[2], 0.05)
        self.assertAlmostEqual(adj[3], 0.05)

    def test_bh_is_never_larger_than_holm(self):
        rng = random.Random(9)
        for _ in range(20):
            ps = [rng.random() for _ in range(rng.randint(1, 10))]
            for a, b in zip(benjamini_hochberg_adjust(ps), holm_adjust(ps)):
                self.assertLessEqual(a, b + 1e-12)

    def test_bh_step_up_decision_matches_definition(self):
        # BH rejects the largest i with p_(i) <= i/m * alpha and everything
        # below it; adjusted-p < alpha must agree.
        rng = random.Random(13)
        for _ in range(30):
            ps = [rng.random() for _ in range(rng.randint(1, 8))]
            alpha = 0.05
            m = len(ps)
            adj = benjamini_hochberg_adjust(ps)
            order = sorted(range(m), key=lambda i: ps[i])
            k = 0
            for rank in range(1, m + 1):
                if ps[order[rank - 1]] <= rank / m * alpha:
                    k = rank
            expected = set(order[:k])
            got = {i for i in range(m) if adj[i] < alpha}
            self.assertEqual(got, expected, msg=f"ps={ps}")

    def test_input_order_preserved(self):
        ps = [0.9, 0.01, 0.5]
        adj = benjamini_hochberg_adjust(ps)
        self.assertAlmostEqual(adj[0], 0.9)
        self.assertAlmostEqual(adj[1], 0.03)
        self.assertAlmostEqual(adj[2], 0.75)


class TestCorrectPvalues(unittest.TestCase):
    def test_dispatch(self):
        r = correct_pvalues([0.01, 0.02], "holm")
        self.assertEqual(r["method"], "holm")
        self.assertIn("FWER", r["controls"])
        r = correct_pvalues([0.01, 0.02], "bh")
        self.assertEqual(r["method"], "benjamini_hochberg")
        self.assertIn("FDR", r["controls"])
        r = correct_pvalues([0.3, 0.7], None)
        self.assertEqual(r["adjusted"], [0.3, 0.7])
        with self.assertRaises(ValueError):
            correct_pvalues([0.1], "bonferroni_deluxe")

    def test_none_agrees_with_uncorrected(self):
        self.assertEqual(correct_pvalues([0.2, 0.4], "none")["adjusted"],
                         benjamini_hochberg_adjust.__self__ if False else [0.2, 0.4])


# ---------------------------------------------------------------------------
# hierarchical bootstrap


class TestHierarchicalBootstrap(unittest.TestCase):
    def _data(self):
        return {
            "tier_a": {"t1": 1.0, "t2": 0.0, "t3": 1.0},
            "tier_b": {"t4": 0.5, "t5": 0.0},
        }

    def test_point_estimate_is_full_data_mean(self):
        e = hierarchical_bootstrap(self._data(), n_boot=50)
        self.assertAlmostEqual(e.value, 2.5 / 5)

    def test_deterministic_given_seed(self):
        a = hierarchical_bootstrap(self._data(), n_boot=40, seed=1)
        b = hierarchical_bootstrap(self._data(), n_boot=40, seed=1)
        self.assertEqual(a.ci, b.ci)
        c = hierarchical_bootstrap(self._data(), n_boot=40, seed=2)
        self.assertNotEqual(a.ci, c.ci)

    def test_list_values_averaged_as_fixed_runs(self):
        # runs within a task are aggregated to their mean and then held fixed
        data = {"s": {"t1": [1.0, 0.0], "t2": [0.0, 0.0]}}
        e = hierarchical_bootstrap(data, n_boot=10)
        self.assertAlmostEqual(e.value, 0.25)

    def test_ci_is_ordered_and_inside_unit_interval(self):
        e = hierarchical_bootstrap(self._data(), n_boot=200, seed=0)
        lo, hi = e.ci
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_replicate_sample_size_follows_sampled_scenarios(self):
        # Structure fixed: 2 scenarios with 3 and 2 tasks -> every
        # replicate draws 2 scenarios, so its size is 4, 5 or 6 depending
        # on which were chosen -- never anything else.
        data = self._data()
        sizes = set()

        def stat(values):
            sizes.add(len(values))
            return sum(values) / len(values)

        hierarchical_bootstrap(data, n_boot=30, seed=0, statistic=stat)
        self.assertEqual(sizes, {4, 5, 6})

    def test_degenerate_single_task_scenario_wide_ci(self):
        data = {"s1": {"t1": 1.0}, "s2": {"t2": 0.0}}
        e = hierarchical_bootstrap(data, n_boot=100, seed=0)
        lo, hi = e.ci
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1.0)  # resampling can pick either task

    def test_validation(self):
        with self.assertRaises(ValueError):
            hierarchical_bootstrap({})
        with self.assertRaises(ValueError):
            hierarchical_bootstrap({"s": {}})
        with self.assertRaises(ValueError):
            hierarchical_bootstrap(self._data(), n_boot=0)
        with self.assertRaises(ValueError):
            hierarchical_bootstrap(self._data(), alpha=0.0)


# ---------------------------------------------------------------------------
# Newcombe hybrid-score interval (paired)


class TestNewcombePaired(unittest.TestCase):
    def test_zero_discordants_gives_symmetric_zero_centered_interval(self):
        # a=10, d=2, b=c=0: p1 == p2 so d_hat = 0.  phi = 1 and the
        # marginals share one Wilson interval, so the interval is exactly
        # symmetric about 0 and contains it (width reflects n=12
        # uncertainty -- the Wilson interval is asymmetric about p_hat).
        lo, hi = newcombe_paired(10, 0, 0, 2)
        self.assertAlmostEqual(hi, -lo, places=12)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_contains_point_estimate(self):
        rng = random.Random(21)
        for _ in range(50):
            a, b, c, d = (rng.randint(0, 30) for _ in range(4))
            if a + b + c + d == 0:
                continue
            lo, hi = newcombe_paired(a, b, c, d)
            d_hat = (b - c) / (a + b + c + d)
            self.assertLessEqual(lo, d_hat + 1e-9, msg=f"{a},{b},{c},{d}")
            self.assertGreaterEqual(hi, d_hat - 1e-9, msg=f"{a},{b},{c},{d}")

    def test_swapping_versions_reflects_interval(self):
        a, b, c, d = 40, 9, 2, 20
        lo1, hi1 = newcombe_paired(a, b, c, d)
        lo2, hi2 = newcombe_paired(a, c, b, d)
        self.assertAlmostEqual(lo1, -hi2, places=9)
        self.assertAlmostEqual(hi1, -lo2, places=9)

    def test_hand_computed_anchor(self):
        # Independent re-derivation of the closed form via wilson_ci:
        a, b, c, d = 40, 9, 2, 20
        n = 71
        p1, p2 = (a + b) / n, (a + c) / n
        l1, u1 = wilson_ci(a + b, n)
        l2, u2 = wilson_ci(a + c, n)
        phi = (a * d - b * c) / math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        d_hat = p1 - p2
        lo_exp = d_hat - math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2
                                       - 2 * phi * (p1 - l1) * (u2 - p2)))
        hi_exp = d_hat + math.sqrt(max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2
                                       - 2 * phi * (u1 - p1) * (p2 - l2)))
        lo, hi = newcombe_paired(a, b, c, d)
        self.assertAlmostEqual(lo, lo_exp, places=12)
        self.assertAlmostEqual(hi, hi_exp, places=12)

    def test_all_improvements_is_positive_interval(self):
        # b=10 improvements only: p1 - p2 = 10/n > 0, lower bound > 0
        lo, hi = newcombe_paired(40, 10, 0, 21)
        self.assertGreater(lo, 0.0)

    def test_margin_at_boundary(self):
        # (0, 5, 0, 5): p1 = 0.5, p2 = 0.0 -> d_hat = 0.5; the interval
        # contains the point estimate and is (mostly) positive.
        lo, hi = newcombe_paired(0, 5, 0, 5)
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)
        self.assertGreater(hi, 0.0)

    def test_perfect_negative_correlation_uses_full_phi(self):
        # (0, 5, 5, 0): every pair flips -> phi = -1, d_hat = 0.  With
        # identical margins the interval is symmetric: [-2x, 2x] where
        # x = u1 - p1 = p2 - l2 from the shared Wilson interval.
        l, u = wilson_ci(5, 10)
        lo, hi = newcombe_paired(0, 5, 5, 0)
        self.assertAlmostEqual(hi, 2.0 * (u - 0.5), places=12)
        self.assertAlmostEqual(lo, -2.0 * (u - 0.5), places=12)

    def test_validation(self):
        with self.assertRaises(ValueError):
            newcombe_paired(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            newcombe_paired(-1, 5, 0, 5)

    def test_interval_agrees_with_midp_direction_often(self):
        # Coverage-ordering sanity: when the mid-p McNemar is significant,
        # the Newcombe interval should usually exclude 0 (small counts keep
        # both honest, so only require agreement in the strong direction).
        from reliagate.compare import mcnemar_midp
        rng = random.Random(31)
        agree = 0
        checked = 0
        for _ in range(200):
            b, c = rng.randint(0, 20), rng.randint(0, 20)
            if b + c < 5:
                continue
            a, d = rng.randint(5, 40), rng.randint(5, 40)
            lo, hi = newcombe_paired(a, b, c, d)
            excl0 = lo > 0 or hi < 0
            sig = mcnemar_midp(b, c) < 0.05
            checked += 1
            if (not sig) or excl0:
                agree += 1
        self.assertEqual(agree, checked)


if __name__ == "__main__":
    unittest.main()
