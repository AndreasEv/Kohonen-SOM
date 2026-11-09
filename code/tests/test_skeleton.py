"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Smoke tests for the project skeleton: the sample parameters.txt has every
setting the assignment requires, and main.py finds it.
"""

import os
import tempfile
import unittest

import main

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_KEYS = {
    "gridRows", "gridCols", "numInputs", "maxIterations",
    "initialLearningRate", "finalLearningRate", "initialRadius",
    "randomSeed", "initMethod", "trainFile", "testFile",
}


class TestSkeleton(unittest.TestCase):
    def test_sample_parameters_has_required_keys(self):
        with open(os.path.join(CODE_DIR, "parameters.txt")) as f:
            keys = {line.split()[0] for line in f if line.strip()}
        self.assertTrue(REQUIRED_KEYS <= keys, REQUIRED_KEYS - keys)

    def test_find_parameters_file(self):
        self.assertIsNotNone(main.find_parameters_file(CODE_DIR))
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(main.find_parameters_file(empty))

    def test_numpy_available(self):
        import numpy  # noqa: F401  (the only required runtime library)


if __name__ == "__main__":
    unittest.main()
