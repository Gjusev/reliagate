"""Gate logic + CLI exit codes (0 pass / 2 regression / 3 infra error)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from reliagate.gate import EXIT_ERROR, EXIT_PASS, EXIT_REGRESSION, gate

from .test_estimators_property import make_run


def write_jsonl(path, data):
    with open(path, "w") as fh:
        for task_id, outcomes in data.items():
            for attempt, s in enumerate(outcomes):
                fh.write(json.dumps(
                    {"task_id": task_id, "attempt": attempt, "success": s}
                ) + "\n")


class TestGateFunction(unittest.TestCase):
    def test_pass_when_no_change(self):
        base = make_run({f"t{i}": [True, True] for i in range(5)})
        new = make_run({f"t{i}": [True, True] for i in range(5)})
        report = gate(base, new)
        self.assertEqual(report.exit_code, EXIT_PASS)
        self.assertEqual(report.decision, "pass")

    def test_regression_exit_2(self):
        # 8 consistent -> 1 consistent: exact McNemar p = 2/256 < 0.05
        base = make_run({f"t{i}": [True, True] for i in range(8)})
        new = make_run({
            **{f"t{i}": [True, True] for i in range(1)},
            **{f"t{i}": [True, False] for i in range(1, 8)},
        })
        report = gate(base, new, max_regression=0.05)
        self.assertEqual(report.exit_code, EXIT_REGRESSION)
        self.assertEqual(report.decision, "regression")
        self.assertIn("significant", report.reason)

    def test_big_drop_but_not_significant_stays_0(self):
        # 2 of 6 tasks lost: exact p = 2*(1+6)/64 > 0.05 -> not significant
        base = make_run({f"t{i}": [True, True] for i in range(6)})
        new = make_run({
            **{f"t{i}": [True, True] for i in range(4)},
            **{f"t{i}": [True, False] for i in range(4, 6)},
        })
        report = gate(base, new)
        self.assertEqual(report.exit_code, EXIT_PASS)
        self.assertIn("NOT statistically significant", report.reason)

    def test_significant_but_within_margin_stays_0(self):
        # 2 tasks lost out of 40: p = 2*(1+8+28+56)/2^8 < 0.05 but drop
        # (0.05) is exactly at the allowed margin -> pass
        base = make_run({f"t{i}": [True, True] for i in range(40)})
        new = make_run({
            **{f"t{i}": [True, True] for i in range(38)},
            **{f"t{i}": [True, False] for i in range(38, 40)},
        })
        report = gate(base, new, max_regression=0.06)
        self.assertEqual(report.exit_code, EXIT_PASS)

    def test_mismatched_task_sets_exit_3(self):
        base = make_run({"a": [True, True], "b": [True, True]})
        new = make_run({"a": [True, True], "z": [True, True]})
        report = gate(base, new)
        self.assertEqual(report.exit_code, EXIT_ERROR)
        self.assertIn("task sets", report.reason)

    def test_bad_alpha_exit_3(self):
        base = make_run({"a": [True]})
        new = make_run({"a": [True]})
        self.assertEqual(gate(base, new, alpha=0.0).exit_code, EXIT_ERROR)

    def test_report_includes_abstention_treatment(self):
        base = make_run({"a": [True, True]})
        new = make_run({"a": [True, None]})
        report = gate(base, new)
        self.assertEqual(report.exit_code, EXIT_PASS)
        self.assertEqual(
            report.abstention["new"]["trials_abstained"], 1
        )
        self.assertIn("never", report.abstention["new"]["treatment"])

    def test_to_dict_roundtrip(self):
        base = make_run({"a": [True, True], "b": [True, True]})
        new = make_run({"a": [True, False], "b": [True, True]})
        d = gate(base, new).to_dict()
        self.assertEqual(d["decision"], "pass")
        self.assertEqual(d["comparison"]["c_lost"], 1)
        self.assertEqual(d["comparison"]["b_gained"], 0)


class TestGateCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base_path = os.path.join(self.tmp.name, "base.jsonl")
        self.new_path = os.path.join(self.tmp.name, "new.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "reliagate", "gate", *args],
            capture_output=True, text=True, cwd="/root/reliagate",
        )

    def test_exit_0_on_pass(self):
        write_jsonl(self.base_path, {"a": [True, True], "b": [True, True]})
        write_jsonl(self.new_path, {"a": [True, True], "b": [True, True]})
        proc = self._run_cli("--base", self.base_path, "--new", self.new_path)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_exit_2_on_significant_regression(self):
        write_jsonl(self.base_path, {f"t{i}": [True, True] for i in range(8)})
        write_jsonl(self.new_path, {
            **{"t0": [True, True]},
            **{f"t{i}": [True, False] for i in range(1, 8)},
        })
        proc = self._run_cli("--base", self.base_path, "--new", self.new_path)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("regression", proc.stdout)

    def test_exit_3_on_malformed_jsonl(self):
        write_jsonl(self.base_path, {"a": [True]})
        with open(self.new_path, "w") as fh:
            fh.write("this is not json\n")
        proc = self._run_cli("--base", self.base_path, "--new", self.new_path)
        self.assertEqual(proc.returncode, 3)
        self.assertIn("infra error", proc.stderr)

    def test_exit_3_on_missing_file(self):
        write_jsonl(self.base_path, {"a": [True]})
        proc = self._run_cli("--base", self.base_path,
                             "--new", "/nonexistent/x.jsonl")
        self.assertEqual(proc.returncode, 3)

    def test_exit_3_on_mismatched_tasks(self):
        write_jsonl(self.base_path, {"a": [True]})
        write_jsonl(self.new_path, {"b": [True]})
        proc = self._run_cli("--base", self.base_path, "--new", self.new_path)
        self.assertEqual(proc.returncode, 3)

    def test_exit_3_on_usage_error_not_2(self):
        # usage errors must be 3 so that exit 2 means exactly one thing
        proc = self._run_cli("--base", self.base_path)  # missing --new
        self.assertEqual(proc.returncode, 3)

    def test_json_output_flag(self):
        write_jsonl(self.base_path, {"a": [True, True]})
        write_jsonl(self.new_path, {"a": [True, True]})
        proc = self._run_cli("--base", self.base_path, "--new", self.new_path,
                             "--json")
        self.assertEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("abstention", payload)


if __name__ == "__main__":
    unittest.main()
