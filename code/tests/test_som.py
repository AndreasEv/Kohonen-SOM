"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Tests for som.py: the square grid layout, both weight initializations
(random and data-sample, reproducible with the seed), and the winner search
by Euclidean distance, checked against a plain loop.
"""

import os
import unittest

import numpy as np

from data import load_train_test
from parameters import read_parameters
from som import (SOM, find_winner, find_winners, grid_positions, init_random_weights,
                 init_sample_weights, init_weights)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestGrid(unittest.TestCase):
    def test_positions_row_major(self):
        np.testing.assert_array_equal(
            grid_positions(2, 3), [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])

    def test_shape_and_unique(self):
        pos = grid_positions(4, 7)
        self.assertEqual(pos.shape, (28, 2))
        self.assertEqual(len({tuple(p) for p in pos}), 28)
        self.assertEqual(pos[:, 0].max(), 3)
        self.assertEqual(pos[:, 1].max(), 6)

    def test_single_neuron(self):
        np.testing.assert_array_equal(grid_positions(1, 1), [[0, 0]])

    def test_index_and_position_round_trip(self):
        som = SOM(3, 5, 2, seed=0)
        for k in range(som.num_neurons):
            self.assertEqual(som.index(*som.position(k)), k)
            self.assertEqual(som.position(k), tuple(som.positions[k]))
        self.assertEqual(som.index(2, 4), 14)

    def test_index_outside_grid(self):
        som = SOM(3, 5, 2, seed=0)
        for row, col in ((3, 0), (0, 5), (-1, 0)):
            with self.subTest(row=row, col=col):
                with self.assertRaises(IndexError):
                    som.index(row, col)

    def test_square_neighbours_are_one_apart(self):
        # On a square grid the 4 direct neighbours of a neuron are at distance 1.
        som = SOM(5, 5, 2, seed=0)
        centre = som.positions[som.index(2, 2)]
        dist = np.linalg.norm(som.positions - centre, axis=1)
        neighbours = {som.position(k) for k in np.flatnonzero(dist == 1)}
        self.assertEqual(neighbours, {(1, 2), (3, 2), (2, 1), (2, 3)})

    def test_bad_size(self):
        for args in ((0, 5, 2), (5, 0, 2), (5, 5, 0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    SOM(*args, seed=0)


class TestInit(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(7).random((50, 4))

    def test_random_shape_and_range(self):
        w = init_random_weights(100, 4, np.random.default_rng(1))
        self.assertEqual(w.shape, (100, 4))
        self.assertGreaterEqual(w.min(), 0.0)
        self.assertLess(w.max(), 1.0)

    def test_random_is_spread_over_unit_cube(self):
        w = init_random_weights(2000, 3, np.random.default_rng(1))
        np.testing.assert_allclose(w.mean(axis=0), 0.5, atol=0.03)

    def test_sample_rows_are_distinct_training_patterns(self):
        w = init_sample_weights(30, self.x, np.random.default_rng(1))
        self.assertEqual(w.shape, (30, 4))
        rows = {tuple(r) for r in self.x}
        self.assertTrue(all(tuple(r) in rows for r in w))
        self.assertEqual(len({tuple(r) for r in w}), 30)  # no repeats

    def test_sample_more_neurons_than_patterns(self):
        w = init_sample_weights(80, self.x, np.random.default_rng(1))
        self.assertEqual(w.shape, (80, 4))
        rows = {tuple(r) for r in self.x}
        self.assertTrue(all(tuple(r) in rows for r in w))

    def test_sample_is_a_copy(self):
        w = init_sample_weights(10, self.x, np.random.default_rng(1))
        before = self.x.copy()
        w += 1
        np.testing.assert_array_equal(self.x, before)

    def test_same_seed_same_weights(self):
        for method in ("random", "sample"):
            with self.subTest(method=method):
                a = SOM(4, 4, 4, seed=42, init_method=method, x_train=self.x)
                b = SOM(4, 4, 4, seed=42, init_method=method, x_train=self.x)
                c = SOM(4, 4, 4, seed=43, init_method=method, x_train=self.x)
                np.testing.assert_array_equal(a.weights, b.weights)
                self.assertFalse(np.array_equal(a.weights, c.weights))

    def test_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "unknown initMethod 'pca'"):
            init_weights("pca", 4, 4, np.random.default_rng(0))

    def test_sample_needs_data(self):
        with self.assertRaisesRegex(ValueError, "needs the training data"):
            SOM(2, 2, 4, seed=0, init_method="sample")

    def test_sample_data_wrong_width(self):
        with self.assertRaisesRegex(ValueError, "must have 3 columns"):
            SOM(2, 2, 3, seed=0, init_method="sample", x_train=self.x)

    def test_sample_empty_data(self):
        with self.assertRaisesRegex(ValueError, "at least one training pattern"):
            init_sample_weights(4, np.empty((0, 4)), np.random.default_rng(0))


class TestWinner(unittest.TestCase):
    def test_hand_example(self):
        w = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        j, d = find_winner(w, [0.9, 0.2])
        self.assertEqual(j, 1)
        self.assertAlmostEqual(d, np.sqrt(0.1 ** 2 + 0.2 ** 2))

    def test_exact_match_has_distance_zero(self):
        w = np.random.default_rng(3).random((9, 5))
        j, d = find_winner(w, w[6])
        self.assertEqual((j, d), (6, 0.0))

    def test_tie_goes_to_lowest_index(self):
        w = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.assertEqual(find_winner(w, [0.0, 0.0])[0], 0)
        idx, _ = find_winners(w, np.zeros((3, 2)))
        np.testing.assert_array_equal(idx, [0, 0, 0])

    def test_batch_matches_loop(self):
        rng = np.random.default_rng(5)
        w = rng.random((37, 6))
        x = rng.random((600, 6))  # more than one chunk
        idx, dist = find_winners(w, x)
        for i, xi in enumerate(x):
            j = int(np.argmin([np.sqrt(np.sum((xi - wj) ** 2)) for wj in w]))
            self.assertEqual(idx[i], j)
            self.assertAlmostEqual(dist[i], np.linalg.norm(xi - w[j]))

    def test_batch_empty(self):
        idx, dist = find_winners(np.ones((3, 2)), np.empty((0, 2)))
        self.assertEqual(len(idx), 0)
        self.assertEqual(len(dist), 0)

    def test_som_methods(self):
        som = SOM(2, 3, 2, seed=0)
        som.weights = grid_positions(2, 3).astype(float)  # weights = grid coordinates
        j, d = som.winner([1.1, 1.8])
        self.assertEqual(j, 5)
        self.assertAlmostEqual(d, np.hypot(0.1, 0.2))
        idx, _ = som.winners(np.array([[0.0, 0.1], [0.9, 0.9]]))
        np.testing.assert_array_equal(idx, [0, 4])


class TestRealData(unittest.TestCase):
    """The SOM builds from the sample parameters.txt and the real letter data."""

    @classmethod
    def setUpClass(cls):
        cls.params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
        cls.data = load_train_test(cls.params, CODE_DIR)

    def test_from_parameters(self):
        p = self.params
        som = SOM.from_parameters(p, self.data.x_train)
        self.assertEqual(som.weights.shape, (p["gridRows"] * p["gridCols"], p["numInputs"]))
        self.assertGreaterEqual(som.weights.min(), 0.0)
        self.assertLessEqual(som.weights.max(), 1.0)

    def test_winners_for_test_set(self):
        som = SOM.from_parameters(self.params, self.data.x_train)
        idx, dist = som.winners(self.data.x_test)
        self.assertEqual(idx.shape, (4000,))
        self.assertTrue(np.all((idx >= 0) & (idx < som.num_neurons)))
        self.assertTrue(np.all(dist >= 0))

    def test_sample_init_starts_closer_than_random(self):
        # Neurons copied from the data start nearer to the data than uniform
        # random ones, so the initial quantization error is lower.
        x = self.data.x_train
        errors = {}
        for method in ("random", "sample"):
            som = SOM(self.params["gridRows"], self.params["gridCols"], 16,
                      seed=42, init_method=method, x_train=x)
            errors[method] = som.winners(x)[1].mean()
        self.assertLess(errors["sample"], errors["random"])


if __name__ == "__main__":
    unittest.main()
