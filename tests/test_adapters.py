"""Adapters: JSONL round-trip, runner budget/concurrency, nitelos mapping."""

import json
import os
import tempfile
import threading
import time
import unittest

from reliagate.adapters.jsonl import JSONLError, read_jsonl, write_jsonl
from reliagate.adapters.nitelos import verdicts_to_runs
from reliagate.adapters.runner import (
    ZAI_BASE_URL,
    ZAI_KEY_ENV,
    ZAI_MODEL,
    run_agent,
)
from reliagate.model import EvalRun, Run

from .test_estimators_property import make_run


class TestJsonl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip(self):
        original = EvalRun(name="v1", runs=[
            Run("a", 0, True, "v1", {"latency_ms": 10}),
            Run("a", 1, False),
            Run("b", 0, None, None, {"err": "timeout"}),
        ])
        self.assertEqual(write_jsonl(original, self.path), 3)
        back = read_jsonl(self.path, name="back")
        self.assertEqual(back.name, "back")
        self.assertEqual(len(back.runs), 3)
        by_key = {(r.task_id, r.attempt): r for r in back.runs}
        self.assertIs(by_key[("a", 0)].success, True)
        self.assertIs(by_key[("a", 1)].success, False)
        self.assertIsNone(by_key[("b", 0)].success)
        self.assertEqual(by_key[("b", 0)].metadata, {"err": "timeout"})
        self.assertEqual(by_key[("a", 0)].metadata, {"latency_ms": 10})

    def test_malformed_line_raises_jsonlerror(self):
        with open(self.path, "w") as fh:
            fh.write("{not json}\n")
        with self.assertRaises(JSONLError):
            read_jsonl(self.path)

    def test_missing_keys_raises(self):
        with open(self.path, "w") as fh:
            fh.write(json.dumps({"task_id": "a", "success": True}) + "\n")
        with self.assertRaises(JSONLError):
            read_jsonl(self.path)

    def test_bad_success_type_raises(self):
        with open(self.path, "w") as fh:
            fh.write(json.dumps({"task_id": "a", "attempt": 0,
                                 "success": "yes"}) + "\n")
        with self.assertRaises(JSONLError):
            read_jsonl(self.path)

    def test_empty_file_raises(self):
        open(self.path, "w").close()
        with self.assertRaises(JSONLError):
            read_jsonl(self.path)

    def test_missing_file_raises(self):
        with self.assertRaises(JSONLError):
            read_jsonl("/no/such/file.jsonl")

    def test_null_success_is_abstained_not_failure(self):
        with open(self.path, "w") as fh:
            fh.write(json.dumps({"task_id": "a", "attempt": 0,
                                 "success": None}) + "\n")
        run = read_jsonl(self.path)
        self.assertIsNone(run.runs[0].success)


class TestRunner(unittest.TestCase):
    def test_basic_k_attempts(self):
        calls = []

        def fn(task):
            calls.append(task)
            return True

        report = run_agent(fn, ["a", "b"], k=3, max_workers=2)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(report.run.runs), 6)
        self.assertEqual(report.attempts_run, 6)
        attempts = sorted(r.attempt for r in report.run.runs
                          if r.task_id == "a")
        self.assertEqual(attempts, [0, 1, 2])

    def test_budget_skips_not_fails(self):
        def fn(task):
            return True, 10.0  # costs 10 per attempt

        report = run_agent(fn, ["a", "b", "c", "d", "e", "f"], k=2,
                           cost_budget=25.0, estimated_cost_per_call=10.0,
                           ordered=True)
        # 25 // 10 = 2 attempts launched, rest skipped as abstained
        self.assertLessEqual(report.cost_spent, 25.0)
        self.assertEqual(report.attempts_run, 2)
        self.assertEqual(report.attempts_skipped_budget, 10)
        skipped = [r for r in report.run.runs
                   if r.metadata.get("reason") == "budget_exhausted"]
        self.assertEqual(len(skipped), 10)
        for r in skipped:
            self.assertIsNone(r.success)  # abstained, NEVER failure

    def test_abstain_propagates_as_none(self):
        report = run_agent(lambda t: None, ["a"], k=1)
        self.assertIsNone(report.run.runs[0].success)

    def test_concurrency_actually_parallel(self):
        barrier = threading.Barrier(4, timeout=5)

        def fn(task):
            barrier.wait()  # deadlocks unless 4 run at once
            return True

        report = run_agent(fn, ["a", "b", "c", "d"], k=1, max_workers=4)
        self.assertEqual(len(report.run.runs), 4)

    def test_validates_args(self):
        with self.assertRaises(ValueError):
            run_agent(lambda t: True, ["a"], k=0)
        with self.assertRaises(ValueError):
            run_agent(lambda t: True, ["a"], max_workers=0)


class TestZaiEndpointConfig(unittest.TestCase):
    def test_pinned_endpoint_and_model(self):
        # owner rule: Z.ai Coding Plan, glm-5.3-flash ONLY
        self.assertEqual(ZAI_BASE_URL, "https://api.z.ai/api/coding/paas/v4")
        self.assertEqual(ZAI_MODEL, "glm-5.3-flash")
        self.assertEqual(ZAI_KEY_ENV, "RELIAGATE_ZAI_KEY")

    def test_no_key_means_no_live_call(self):
        from reliagate.adapters.runner import make_zai_agent_fn
        with self.assertRaises(RuntimeError) as ctx:
            make_zai_agent_fn(lambda t: t)
        self.assertIn("opt-in", str(ctx.exception))
        self.assertIn("RELIAGATE_ZAI_KEY", str(ctx.exception))

    def test_other_models_rejected(self):
        from reliagate.adapters.runner import make_zai_agent_fn
        os.environ[ZAI_KEY_ENV] = "dummy-for-test"
        try:
            with self.assertRaises(ValueError) as ctx:
                make_zai_agent_fn(lambda t: t, model="some-other-model")
            self.assertIn("glm-5.3-flash", str(ctx.exception))
        finally:
            del os.environ[ZAI_KEY_ENV]


class TestNitelos(unittest.TestCase):
    def test_string_verdicts_map(self):
        runs = verdicts_to_runs(
            ["pass", "failed", "error", None],
            task_ids=["a", "a", "b", "b"],
        )
        by_key = {(r.task_id, r.attempt): r.success for r in runs}
        self.assertIs(by_key[("a", 0)], True)
        self.assertIs(by_key[("a", 1)], False)
        self.assertIsNone(by_key[("b", 0)])
        self.assertIsNone(by_key[("b", 1)])

    def test_dict_verdicts_carry_task_id(self):
        runs = verdicts_to_runs([
            {"task_id": "x", "verdict": "pass"},
            {"task_id": "x", "verdict": "fail"},
        ])
        self.assertEqual([r.success for r in runs], [True, False])
        self.assertEqual([r.attempt for r in runs], [0, 1])

    def test_object_verdicts(self):
        class Verdict:
            task_id = "y"
            passed = True

        runs = verdicts_to_runs([Verdict()])
        self.assertEqual(runs[0].task_id, "y")
        self.assertIs(runs[0].success, True)

    def test_requires_task_ids_when_missing(self):
        with self.assertRaises(ValueError):
            verdicts_to_runs(["pass"])


if __name__ == "__main__":
    unittest.main()
