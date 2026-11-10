"""Offline regression checks for the cvc5 Python-API benchmark oracles.

Run with: python -m unittest benchmarks.test_smt
"""

import json
import unittest
from unittest.mock import patch

from tasks.benchmarks import (
    BenchmarkTask,
    _crypto_smt,
    _list_sequence_smt,
    _smt_choice,
)


class BenchmarkSMTTests(unittest.TestCase):
    def test_choice_returns_distinct_bounded_witnesses(self):
        self.assertEqual(set(_smt_choice([2, 5, 9], 8)), {2, 5, 9})
        self.assertEqual(len(_smt_choice([2, 5, 9], 2)), 2)
        self.assertEqual(_smt_choice([], 8), [])
        self.assertEqual(_smt_choice([2], 0), [])

    def test_crypto_equivalence_without_subprocess(self):
        # A missing CLI must not prevent the task from proving equality.
        task = BenchmarkTask("cryptography", "caesar:0")
        with patch("subprocess.run", side_effect=FileNotFoundError("no CLI")):
            self.assertEqual(task.counterexamples(
                json.dumps({"cipher": "caesar", "shift": 3})), ([], True))
        self.assertEqual(task.score(json.dumps({"cipher": "caesar", "shift": 3})), 1.0)

    def test_crypto_witnesses_are_real_and_unique(self):
        reference = {"cipher": "caesar", "shift": 3}
        candidate = {"cipher": "caesar", "shift": 4}
        witnesses, equivalent = _crypto_smt(reference, candidate, 60)
        self.assertFalse(equivalent)
        self.assertEqual(len(witnesses), 52)
        self.assertEqual(len(set(witnesses)), 52)
        for ch in witnesses:
            self.assertNotEqual(BenchmarkTask._apply_cipher(reference, ch),
                                BenchmarkTask._apply_cipher(candidate, ch))

    def test_crypto_equivalent_shift_wraparound(self):
        self.assertEqual(_crypto_smt(
            {"cipher": "caesar", "shift": 3},
            {"cipher": "caesar", "shift": 29}, 8), ([], True))

    def test_list_bounded_identity_has_no_witness(self):
        self.assertIsNone(_list_sequence_smt(
            "def reference(x):\n    return x",
            "def transform(x):\n    return x[:]"))

    def test_list_reverse_witness_is_real(self):
        witness = _list_sequence_smt(
            "def reference(x):\n    return x",
            "def transform(x):\n    return x[::-1]")
        self.assertTrue(3 <= len(witness) <= 8)
        self.assertTrue(all(0 <= value <= 100 for value in witness))
        self.assertNotEqual(witness, witness[::-1])

    def test_list_task_finds_concrete_disagreements_without_cli(self):
        task = BenchmarkTask("list_functions", "c102")
        with patch("subprocess.run", side_effect=FileNotFoundError("no CLI")):
            pairs, equivalent = task.counterexamples(
                "def transform(x):\n    return x[::-1]", limit=3)
        self.assertFalse(equivalent)
        self.assertTrue(1 <= len(pairs) <= 3)
        for x, expected in pairs:
            self.assertEqual(expected, x)
            self.assertNotEqual(x[::-1], expected)


if __name__ == "__main__":
    unittest.main()
