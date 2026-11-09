"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Tests for the training loop (training.py), the quantization error, the
results.txt writer (results.py) and the full program (main.run). The main
check is a small synthetic data set with known clusters: after training,
every cluster must have its own neurons, each neuron must sit close to its
cluster centre, and the error on unseen test points from the same clusters
must be as low as on the training points.
"""

import os
import tempfile
import unittest

import numpy as np

import main
from results import format_results, write_results
from som import SOM, quantization_error, schedule
from training import EpochResult, train

LETTERS = "ABCD"
# Four clusters in the unit square (2 features), far apart from each other.
CENTRES = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])
SPREAD = 0.03

PARAMS = {"maxIterations": 25, "initialLearningRate": 0.5, "finalLearningRate": 0.01,
          "initialRadius": 3.0, "finalRadius": 0.5}


def make_clusters(per_cluster, seed):
    """Synthetic patterns around CENTRES; returns (x, cluster index of each row)."""
    rng = np.random.default_rng(seed)
    x = np.vstack([c + SPREAD * rng.standard_normal((per_cluster, 2)) for c in CENTRES])
    return np.clip(x, 0.0, 1.0), np.repeat(np.arange(len(CENTRES)), per_cluster)


class RecordingSOM(SOM):
    """A SOM that remembers which pattern it was shown at every step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def train_step(self, x, learning_rate, radius):
        self.seen.append((tuple(x), learning_rate, radius))
        return super().train_step(x, learning_rate, radius)


class TestQuantizationError(unittest.TestCase):
    def test_hand_example(self):
        # Two neurons at 0 and 1 (1 input). Patterns 0.1 -> 0.1, 0.7 -> 0.3,
        # 1.0 -> 0.0. Mean = 0.4 / 3.
        w = np.array([[0.0], [1.0]])
        x = np.array([[0.1], [0.7], [1.0]])
        self.assertAlmostEqual(quantization_error(w, x), 0.4 / 3)

    def test_zero_when_every_pattern_is_a_neuron(self):
        w = np.random.default_rng(0).random((5, 3))
        self.assertAlmostEqual(quantization_error(w, w[[4, 0, 2]]), 0.0)

    def test_euclidean_not_squared(self):
        w = np.array([[0.0, 0.0]])
        self.assertAlmostEqual(quantization_error(w, np.array([[0.3, 0.4]])), 0.5)


class TestEpoch(unittest.TestCase):
    def setUp(self):
        self.x, _ = make_clusters(10, seed=0)

    def test_every_pattern_once_per_epoch_in_new_order(self):
        som = RecordingSOM(3, 3, 2, seed=1)
        train(som, self.x, self.x, dict(PARAMS, maxIterations=3))
        n = len(self.x)
        self.assertEqual(len(som.seen), 3 * n)
        orders = []
        for e in range(3):
            shown = [s[0] for s in som.seen[e * n:(e + 1) * n]]
            self.assertEqual(sorted(shown), sorted(map(tuple, self.x)))   # each pattern once
            orders.append(shown)
        self.assertNotEqual(orders[0], orders[1])                         # reshuffled
        self.assertNotEqual(orders[0], [tuple(r) for r in self.x])        # not file order

    def test_epoch_uses_its_schedule(self):
        som = RecordingSOM(3, 3, 2, seed=1)
        params = dict(PARAMS, maxIterations=4)
        history = train(som, self.x, self.x, params)
        n = len(self.x)
        for e in range(4):
            eta, sigma = schedule(params, e)
            self.assertTrue(all(s[1] == eta and s[2] == sigma
                                for s in som.seen[e * n:(e + 1) * n]))
            self.assertEqual((history[e].learning_rate, history[e].radius), (eta, sigma))

    def test_test_set_never_changes_the_weights(self):
        x_test, _ = make_clusters(10, seed=5)
        a, b = SOM(3, 3, 2, seed=1), SOM(3, 3, 2, seed=1)
        train(a, self.x, x_test, PARAMS)
        train(b, self.x, x_test + 0.5, PARAMS)
        np.testing.assert_array_equal(a.weights, b.weights)


class TestTrainingLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x_train, cls.y_train = make_clusters(40, seed=10)
        cls.x_test, cls.y_test = make_clusters(10, seed=11)
        cls.som = SOM(5, 5, 2, seed=3, init_method="random")
        cls.start_error = cls.som.quantization_error(cls.x_train)
        cls.calls = []
        cls.history = train(cls.som, cls.x_train, cls.x_test, PARAMS,
                            on_epoch=cls.calls.append)

    def test_one_row_per_epoch(self):
        self.assertEqual([r.epoch for r in self.history], list(range(1, 26)))
        self.assertEqual(self.calls, self.history)

    def test_errors_are_measured_after_the_epoch(self):
        last = self.history[-1]
        self.assertAlmostEqual(last.train_error, self.som.quantization_error(self.x_train))
        self.assertAlmostEqual(last.test_error, self.som.quantization_error(self.x_test))

    def test_error_drops_to_cluster_spread(self):
        train_err = [r.train_error for r in self.history]
        self.assertLess(train_err[-1], self.start_error / 3)
        # The points are about 0.03 * sqrt(pi/2) ~ 0.04 from their centre, so a
        # map that learned the clusters has an error of that order.
        self.assertLess(train_err[-1], 0.05)
        # In the fine-tuning half the error only goes down (small noise allowed).
        second_half = np.array(train_err[12:])
        self.assertTrue(np.all(np.diff(second_half) < 1e-3), second_half)

    def test_test_error_close_to_training_error(self):
        last = self.history[-1]
        self.assertLess(abs(last.test_error - last.train_error), 0.02)

    def test_clusters_get_separate_neurons_near_their_centre(self):
        idx, _ = self.som.winners(self.x_train)
        groups = [set(idx[self.y_train == k]) for k in range(len(CENTRES))]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                self.assertFalse(groups[a] & groups[b])
        for k, neurons in enumerate(groups):
            for j in neurons:
                self.assertLess(np.linalg.norm(self.som.weights[j] - CENTRES[k]), 0.15)
        # Test points land on the neurons of their own cluster.
        test_idx, _ = self.som.winners(self.x_test)
        for k in range(len(CENTRES)):
            self.assertTrue(set(test_idx[self.y_test == k]) <= groups[k])

    def test_same_seed_same_run(self):
        other = SOM(5, 5, 2, seed=3, init_method="random")
        again = train(other, self.x_train, self.x_test, PARAMS)
        self.assertEqual(again, self.history)
        different = train(SOM(5, 5, 2, seed=4, init_method="random"),
                          self.x_train, self.x_test, PARAMS)
        self.assertNotEqual(different, self.history)


class TestResultsFile(unittest.TestCase):
    def test_format(self):
        history = [EpochResult(1, 0.5, 3.0, 0.5123456, 0.6), EpochResult(2, 0.4, 2.0, 0.25, 0.3)]
        self.assertEqual(format_results(history),
                         ["1 0.512346 0.600000", "2 0.250000 0.300000"])

    def test_written_file_reads_back_as_three_columns(self):
        history = [EpochResult(e, 0.1, 1.0, 1.0 / e, 2.0 / e) for e in range(1, 6)]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "results.txt")
            write_results(path, history)
            table = np.loadtxt(path)
        self.assertEqual(table.shape, (5, 3))
        np.testing.assert_array_equal(table[:, 0], [1, 2, 3, 4, 5])
        np.testing.assert_allclose(table[:, 1], [1.0 / e for e in range(1, 6)], atol=1e-6)
        np.testing.assert_allclose(table[:, 2], [2.0 / e for e in range(1, 6)], atol=1e-6)


class TestProgram(unittest.TestCase):
    """main.run on a tiny folder: parameters.txt, letter data files with 4 clusters."""

    def test_run_writes_results(self):
        x_train, y_train = make_clusters(20, seed=20)
        x_test, y_test = make_clusters(5, seed=21)
        with tempfile.TemporaryDirectory() as folder:
            for name, x, y in (("tr.txt", x_train, y_train), ("te.txt", x_test, y_test)):
                with open(os.path.join(folder, name), "w") as f:
                    for row, k in zip(np.round(x * 15).astype(int), y):
                        f.write(LETTERS[k] + "," + ",".join(map(str, row)) + "\n")
            with open(os.path.join(folder, "parameters.txt"), "w") as f:
                f.write("gridRows 4\ngridCols 4\nnumInputs 2\nmaxIterations 8\n"
                        "initialLearningRate 0.5\nfinalLearningRate 0.05\n"
                        "initialRadius 2\nfinalRadius 0.5\nrandomSeed 7\n"
                        "initMethod sample\ntrainFile tr.txt\ntestFile te.txt\n")
            _, _, history = main.run(folder, verbose=False)
            table = np.loadtxt(os.path.join(folder, "results.txt"))
        self.assertEqual(table.shape, (8, 3))
        np.testing.assert_array_equal(table[:, 0], np.arange(1, 9))
        np.testing.assert_allclose(table[:, 1], [r.train_error for r in history], atol=1e-6)
        self.assertLess(table[-1, 1], 0.1)
        self.assertLess(table[-1, 2], 0.1)


if __name__ == "__main__":
    unittest.main()
