"""The README quickstart must run and print what the README says it prints."""

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestReadmeQuickstart(unittest.TestCase):
    def test_quickstart_runs_and_matches_its_comments(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
            readme = fh.read()
        block = re.search(r"## Quickstart.*?```python\n(.*?)```", readme, re.S).group(1)
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            ns: dict = {}
            exec(block, ns)  # noqa: S102 - our own README
        finally:
            os.chdir(cwd)
        v1, v2 = ns["v1"], ns["v2"]
        pass_at_k, pass_k, compare, gate = ns["pass_at_k"], ns["pass_k"], ns["compare"], ns["gate"]
        # the values the README states in its comments
        self.assertEqual(pass_at_k(v2, k=4), 1.0)
        self.assertAlmostEqual(pass_k(v2, k=4).value, 0.65)
        lo, hi = pass_k(v2, k=4, ci="wilson_indicator").ci
        self.assertEqual((round(lo, 3), round(hi, 3)), (0.495, 0.779))
        r = compare(v1, v2)
        self.assertEqual((r.b, r.c, round(r.p_value, 5)), (1, 9, 0.02148))
        self.assertEqual(gate(v1, v2, max_regression=0.05).exit_code, 2)
