"""Offline suite planning checks; no model or network calls."""

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.run_suite import ROOT, main, settings, list_complexity_tiers


class SuiteTests(unittest.TestCase):
    def test_full_catalog_matches_local_settings(self):
        found = settings()
        self.assertEqual(len(found), len(set(found)))
        self.assertEqual({task: sum(name == task for name, _ in found)
                          for task in ("cryptography", "list_functions", "instruction_induction", "scan")},
                         {"cryptography": 300, "list_functions": 250,
                          "instruction_induction": 24, "scan": 8})
        for task, ident in found:
            if task == "scan":
                self.assertTrue((ROOT / "datasets/scan/raw" / f"tasks_test_{ident}.txt").is_file())

    def test_balanced_strata_and_reproducibility(self):
        found = settings("balanced")
        self.assertEqual(len(found), 68)
        self.assertEqual(found, settings("balanced"))
        self.assertTrue(set(found) <= set(settings("all")))
        self.assertNotEqual(found, settings("balanced", sample_seed=44))
        list_ids = {ident for task, ident in found if task == "list_functions"}
        for tier in list_complexity_tiers():
            self.assertEqual(len(list_ids & {ident for _, ident in tier}), 8)
        for family in ("caesar", "atbash", "keyboard"):
            self.assertEqual(sum(task == "cryptography" and ident.startswith(family + ":")
                                 for task, ident in found), 4)
        self.assertEqual(settings("balanced", ["list_functions"]),
                         [pair for pair in found if pair[0] == "list_functions"])

    def test_balanced_custom_size_and_validation(self):
        self.assertEqual(len(settings("balanced", list_per_tier=4, crypto_per_family=2)), 50)
        with self.assertRaises(ValueError):
            settings("balanced", list_per_tier=0)
        with self.assertRaises(ValueError):
            settings("balanced", crypto_per_family=101)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("benchmarks.run_suite.subprocess.run") as run:
            self.assertEqual(main(["icl_gen", "--suite", "balanced", "--dry-run"]), 0)
        run.assert_not_called()
        self.assertIn("Planned 136 jobs", output.getvalue())

    def test_full_dry_run_does_not_launch_processes(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("benchmarks.run_suite.subprocess.run") as run:
            self.assertEqual(main(["both", "--suite", "all", "--dry-run"]), 0)
        run.assert_not_called()
        self.assertIn("jobs=2328", output.getvalue())
        self.assertIn("Planned 2328 jobs", output.getvalue())

    def test_smoke_compatibility_and_extra_options(self):
        with contextlib.redirect_stdout(io.StringIO()), patch(
                "benchmarks.run_suite.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(main(["icl_gen", "--", "--retries", "3"]), 0)
        self.assertEqual(run.call_count, 8)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[-2:], ["--retries", "3"])
            self.assertIn("--rerun", command)

    def test_filter_and_failure_exit(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), patch(
                "benchmarks.run_suite.subprocess.run", return_value=SimpleNamespace(returncode=2)) as run:
            self.assertEqual(main(["icl", "--task", "scan", "--mode", "std", "--keep-going"]), 1)
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("--use_ce", run.call_args.args[0])

    def test_stop_on_first_failure(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), patch(
                "benchmarks.run_suite.subprocess.run", return_value=SimpleNamespace(returncode=2)) as run:
            self.assertEqual(main(["icl_gen"]), 1)
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
