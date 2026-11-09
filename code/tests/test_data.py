"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Tests for data.py: parsing of the letter files, min-max normalization with
training-set statistics only, and the real training/test files.
"""

import os
import tempfile
import unittest

import numpy as np

from data import (DataError, fit_min_max, load_data, load_train_test,
                  normalize, parse_data)
from parameters import read_parameters

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestParse(unittest.TestCase):
    def test_parse_lines(self):
        x, labels = parse_data(["A,1,2,3", "", "b, 4 ,5,6\n"], num_inputs=3)
        np.testing.assert_array_equal(x, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(x.dtype, float)
        np.testing.assert_array_equal(labels, ["A", "B"])

    def test_label_is_not_a_feature(self):
        x, _ = parse_data(["Z,0,0"], num_inputs=2)
        self.assertEqual(x.shape, (1, 2))

    def test_wrong_field_count(self):
        with self.assertRaisesRegex(DataError, "line 2: expected a letter and 3 values, got 3"):
            parse_data(["A,1,2,3", "A,1,2"], num_inputs=3)

    def test_bad_label(self):
        for line in ("1,1,2", "AB,1,2", ",1,2"):
            with self.subTest(line=line):
                with self.assertRaisesRegex(DataError, "first field must be a letter"):
                    parse_data([line], num_inputs=2)

    def test_non_numeric_feature(self):
        with self.assertRaisesRegex(DataError, "features must be numbers"):
            parse_data(["A,1,x"], num_inputs=2)

    def test_non_finite_feature(self):
        with self.assertRaisesRegex(DataError, "finite"):
            parse_data(["A,1,nan"], num_inputs=2)

    def test_empty(self):
        with self.assertRaisesRegex(DataError, "no data"):
            parse_data(["", "  "], num_inputs=2)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(DataError, "file not found"):
                load_data(os.path.join(empty, "training.txt"), 16)


class TestNormalize(unittest.TestCase):
    def test_min_max_to_unit_range(self):
        x = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        lo, hi = fit_min_max(x)
        np.testing.assert_array_equal(lo, [0, 10])
        np.testing.assert_array_equal(hi, [10, 30])
        np.testing.assert_allclose(normalize(x, lo, hi), [[0, 0], [0.5, 0.5], [1, 1]])

    def test_constant_feature_becomes_zero(self):
        x = np.array([[3.0, 1.0], [3.0, 2.0]])
        out = normalize(x, *fit_min_max(x))
        np.testing.assert_array_equal(out[:, 0], [0, 0])
        np.testing.assert_array_equal(out[:, 1], [0, 1])

    def test_uses_given_stats_and_clips(self):
        lo, hi = np.array([2.0]), np.array([6.0])
        x_test = np.array([[0.0], [4.0], [9.0]])
        np.testing.assert_allclose(normalize(x_test, lo, hi), [[0], [0.5], [1]])

    def test_input_not_modified(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        before = x.copy()
        normalize(x, *fit_min_max(x))
        np.testing.assert_array_equal(x, before)


class TestTrainTestSplit(unittest.TestCase):
    """Normalization statistics must come from the training file only."""

    def write(self, folder, name, text):
        with open(os.path.join(folder, name), "w") as f:
            f.write(text)

    def test_test_set_uses_training_stats(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write(folder, "tr.txt", "A,0,10\nB,10,20\n")
            self.write(folder, "te.txt", "A,5,40\n")
            params = {"numInputs": 2, "trainFile": "tr.txt", "testFile": "te.txt"}
            d = load_train_test(params, folder)
            np.testing.assert_allclose(d.x_train, [[0, 0], [1, 1]])
            # 5 is halfway in the training range; 40 is past it and clipped.
            np.testing.assert_allclose(d.x_test, [[0.5, 1.0]])
            np.testing.assert_array_equal(d.feature_max, [10, 20])
            np.testing.assert_array_equal(d.labels_test, ["A"])


class TestRealFiles(unittest.TestCase):
    """The Assignment 2 files shipped in code/ load and normalize correctly."""

    @classmethod
    def setUpClass(cls):
        params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
        cls.d = load_train_test(params, CODE_DIR)

    def test_shapes(self):
        self.assertEqual(self.d.x_train.shape, (16000, 16))
        self.assertEqual(self.d.x_test.shape, (4000, 16))
        self.assertEqual(len(self.d.labels_train), 16000)
        self.assertEqual(len(self.d.labels_test), 4000)

    def test_first_line(self):
        # training.txt starts with "T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8"
        self.assertEqual(self.d.labels_train[0], "T")
        self.assertAlmostEqual(self.d.x_train[0, 0], 2 / 15)

    def test_all_26_letters(self):
        self.assertEqual(len(set(self.d.labels_train)), 26)
        self.assertEqual(len(set(self.d.labels_test)), 26)

    def test_values_in_unit_range(self):
        for x in (self.d.x_train, self.d.x_test):
            self.assertGreaterEqual(x.min(), 0.0)
            self.assertLessEqual(x.max(), 1.0)

    def test_training_set_spans_full_range(self):
        np.testing.assert_allclose(self.d.x_train.min(axis=0), 0.0)
        np.testing.assert_allclose(self.d.x_train.max(axis=0), 1.0)


if __name__ == "__main__":
    unittest.main()
