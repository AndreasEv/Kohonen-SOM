"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Tests for parameters.py: the sample parameters.txt is read with the right
types, and every kind of bad input is rejected with a clear message.
"""

import os
import tempfile
import unittest

from parameters import ParameterError, parse_parameters, read_parameters

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VALID = """\
gridRows 20
gridCols 20
numInputs 16
maxIterations 100
initialLearningRate 0.5
finalLearningRate 0.01
initialRadius 10
randomSeed 42
trainFile training.txt
testFile test.txt
"""


def with_line(name, value):
    """Return VALID with the value of `name` replaced by `value`."""
    lines = [f"{name} {value}" if line.split()[0] == name else line
             for line in VALID.splitlines()]
    return lines


def without(name):
    """Return VALID without the line for `name`."""
    return [line for line in VALID.splitlines() if line.split()[0] != name]


class TestReadSampleFile(unittest.TestCase):
    def test_sample_file_values_and_types(self):
        p = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
        self.assertEqual(p, {
            "gridRows": 20, "gridCols": 20, "numInputs": 16, "maxIterations": 100,
            "initialLearningRate": 0.5, "finalLearningRate": 0.01,
            "initialRadius": 10.0, "randomSeed": 42,
            "trainFile": "training.txt", "testFile": "test.txt",
        })
        self.assertIsInstance(p["gridRows"], int)
        self.assertIsInstance(p["initialRadius"], float)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ParameterError, "file not found"):
                read_parameters(os.path.join(empty, "parameters.txt"))

    def test_read_from_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "parameters.txt")
            with open(path, "w") as f:
                f.write(VALID)
            self.assertEqual(read_parameters(path)["numInputs"], 16)


class TestFormat(unittest.TestCase):
    def test_comments_blank_lines_and_spacing(self):
        lines = ["# SOM settings", ""] + [
            f"  {line.replace(' ', chr(9))}   # comment" for line in VALID.splitlines()]
        self.assertEqual(parse_parameters(lines)["gridCols"], 20)

    def test_any_order(self):
        lines = list(reversed(VALID.splitlines()))
        self.assertEqual(parse_parameters(lines)["randomSeed"], 42)

    def test_missing_value(self):
        lines = without("gridRows") + ["gridRows"]
        with self.assertRaisesRegex(ParameterError, "line 10: expected 'name value'"):
            parse_parameters(lines)

    def test_too_many_values(self):
        with self.assertRaisesRegex(ParameterError, "expected 'name value'"):
            parse_parameters(with_line("trainFile", "my data.txt"))

    def test_unknown_parameter(self):
        with self.assertRaisesRegex(ParameterError, "unknown parameter 'gridrows'"):
            parse_parameters(VALID.splitlines() + ["gridrows 5"])

    def test_duplicate(self):
        with self.assertRaisesRegex(ParameterError, "line 11: gridRows is given more than once"):
            parse_parameters(VALID.splitlines() + ["gridRows 5"])

    def test_missing_parameters_are_all_listed(self):
        lines = [l for l in without("gridRows") if not l.startswith("testFile")]
        with self.assertRaisesRegex(ParameterError, "missing parameter\\(s\\): gridRows, testFile"):
            parse_parameters(lines)

    def test_empty_file(self):
        with self.assertRaisesRegex(ParameterError, "missing parameter"):
            parse_parameters([])


class TestValues(unittest.TestCase):
    def assert_rejected(self, name, value):
        with self.assertRaisesRegex(ParameterError, f"{name} must be"):
            parse_parameters(with_line(name, value))

    def test_integers_reject_bad_values(self):
        for name in ("gridRows", "gridCols", "numInputs", "maxIterations"):
            for value in ("0", "-3", "2.5", "ten"):
                with self.subTest(name=name, value=value):
                    self.assert_rejected(name, value)

    def test_seed(self):
        self.assertEqual(parse_parameters(with_line("randomSeed", "0"))["randomSeed"], 0)
        self.assert_rejected("randomSeed", "-1")
        self.assert_rejected("randomSeed", "1.5")

    def test_learning_rates(self):
        for name in ("initialLearningRate", "finalLearningRate"):
            for value in ("0", "-0.1", "1.5", "nan", "inf", "fast"):
                with self.subTest(name=name, value=value):
                    self.assert_rejected(name, value)
        p = parse_parameters(with_line("initialLearningRate", "1"))
        self.assertEqual(p["initialLearningRate"], 1.0)

    def test_scientific_notation(self):
        p = parse_parameters(with_line("finalLearningRate", "1e-3"))
        self.assertAlmostEqual(p["finalLearningRate"], 0.001)

    def test_final_rate_larger_than_initial(self):
        with self.assertRaisesRegex(ParameterError, "must not be larger than initialLearningRate"):
            parse_parameters(with_line("finalLearningRate", "0.9"))

    def test_equal_rates_allowed(self):
        p = parse_parameters(with_line("finalLearningRate", "0.5"))
        self.assertEqual(p["finalLearningRate"], 0.5)

    def test_radius(self):
        self.assertEqual(parse_parameters(with_line("initialRadius", "2.5"))["initialRadius"], 2.5)
        for value in ("0", "-1", "big"):
            with self.subTest(value=value):
                self.assert_rejected("initialRadius", value)

    def test_error_names_the_line(self):
        with self.assertRaisesRegex(ParameterError, "parameters.txt, line 1: gridRows"):
            parse_parameters(with_line("gridRows", "x"))


if __name__ == "__main__":
    unittest.main()
