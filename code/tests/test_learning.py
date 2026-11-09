"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Tests for the learning part of som.py: the Gaussian neighbourhood on the
grid, the exponential decay of the learning rate and radius, and the
per-pattern weight update, checked against hand-worked numbers, against a
plain loop, and on a small synthetic data set whose final order is known.
"""

import math
import os
import unittest

import numpy as np

from data import load_train_test
from parameters import read_parameters
from som import (SOM, exponential_decay, gaussian_neighbourhood, grid_positions,
                 grid_sq_distances, schedule, update_weights)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestNeighbourhood(unittest.TestCase):
    def test_grid_distances_hand_example(self):
        # 2x3 grid, winner is neuron 4 = (1, 1).
        d2 = grid_sq_distances(grid_positions(2, 3), 4)
        np.testing.assert_array_equal(d2, [2, 1, 2, 1, 0, 1])

    def test_winner_is_one(self):
        h = gaussian_neighbourhood(grid_positions(5, 5), 12, 2.0)
        self.assertEqual(h[12], 1.0)
        self.assertEqual(h.argmax(), 12)

    def test_values_at_known_distances(self):
        # On a 1 x 7 grid the winner 0 has neighbours at distance 0..6.
        sigma = 2.0
        h = gaussian_neighbourhood(grid_positions(1, 7), 0, sigma)
        expected = [math.exp(-d * d / (2 * sigma * sigma)) for d in range(7)]
        np.testing.assert_allclose(h, expected)
        self.assertAlmostEqual(h[2], math.exp(-0.5))    # distance = sigma

    def test_symmetric_and_decreasing(self):
        som = SOM(7, 7, 2, seed=0)
        c = som.index(3, 3)
        h = som.neighbourhood(c, 1.5).reshape(7, 7)
        np.testing.assert_allclose(h, h.T)                  # symmetric about the diagonal
        np.testing.assert_allclose(h, h[::-1, ::-1])        # and about the centre
        row = h[3]
        self.assertTrue(np.all(np.diff(row[3:]) < 0))       # fades with distance

    def test_wider_radius_reaches_further(self):
        pos = grid_positions(10, 10)
        narrow = gaussian_neighbourhood(pos, 0, 1.0)
        wide = gaussian_neighbourhood(pos, 0, 5.0)
        self.assertTrue(np.all(wide[1:] > narrow[1:]))
        self.assertTrue(np.all((wide > 0) & (wide <= 1)))

    def test_bad_radius(self):
        for radius in (0, -1):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    gaussian_neighbourhood(grid_positions(2, 2), 0, radius)


class TestDecay(unittest.TestCase):
    def test_end_points(self):
        self.assertAlmostEqual(exponential_decay(0.5, 0.01, 0, 100), 0.5)
        self.assertAlmostEqual(exponential_decay(0.5, 0.01, 99, 100), 0.01)

    def test_constant_ratio_between_epochs(self):
        values = [exponential_decay(10, 1, t, 11) for t in range(11)]
        ratios = np.array(values[1:]) / np.array(values[:-1])
        np.testing.assert_allclose(ratios, 10 ** (-1 / 10))
        self.assertTrue(np.all(np.diff(values) < 0))

    def test_hand_example(self):
        # Halfway between 0.5 and 0.02 on a log scale is sqrt(0.5 * 0.02) = 0.1.
        self.assertAlmostEqual(exponential_decay(0.5, 0.02, 2, 5), 0.1)

    def test_equal_values_stay_constant(self):
        for t in range(5):
            self.assertAlmostEqual(exponential_decay(0.3, 0.3, t, 5), 0.3)

    def test_single_epoch_uses_initial(self):
        self.assertEqual(exponential_decay(0.5, 0.01, 0, 1), 0.5)

    def test_bad_arguments(self):
        for args in ((0.5, 0.01, 100, 100), (0.5, 0.01, -1, 100),
                     (0.5, 0.01, 0, 0), (0, 0.01, 0, 10), (0.5, 0, 0, 10)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    exponential_decay(*args)

    def test_schedule_from_parameters(self):
        p = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
        last = p["maxIterations"] - 1
        eta0, sigma0 = schedule(p, 0)
        eta_t, sigma_t = schedule(p, last)
        self.assertAlmostEqual(eta0, p["initialLearningRate"])
        self.assertAlmostEqual(sigma0, p["initialRadius"])
        self.assertAlmostEqual(eta_t, p["finalLearningRate"])
        self.assertAlmostEqual(sigma_t, p["finalRadius"])


class TestUpdate(unittest.TestCase):
    def test_hand_example(self):
        # 1 x 3 grid, all weights 0, input x = (1, 2), winner 0, eta 0.5, sigma 1.
        pos = grid_positions(1, 3)
        w = np.zeros((3, 2))
        update_weights(w, pos, [1.0, 2.0], 0, 0.5, 1.0)
        for j, d in enumerate((0, 1, 2)):
            step = 0.5 * math.exp(-d * d / 2.0)
            np.testing.assert_allclose(w[j], [step * 1.0, step * 2.0])

    def test_matches_loop(self):
        rng = np.random.default_rng(3)
        pos = grid_positions(4, 5)
        w = rng.random((20, 6))
        x = rng.random(6)
        expected = w.copy()
        for j in range(20):
            d2 = np.sum((pos[j] - pos[7]) ** 2)
            h = math.exp(-d2 / (2 * 1.7 ** 2))
            expected[j] = expected[j] + 0.3 * h * (x - expected[j])
        update_weights(w, pos, x, 7, 0.3, 1.7)
        np.testing.assert_allclose(w, expected)

    def test_in_place_and_returned(self):
        w = np.zeros((4, 2))
        out = update_weights(w, grid_positions(2, 2), [1.0, 1.0], 0, 0.1, 1.0)
        self.assertIs(out, w)
        self.assertGreater(w[0, 0], 0)

    def test_everyone_moves_closer_winner_most(self):
        rng = np.random.default_rng(8)
        pos = grid_positions(6, 6)
        w = rng.random((36, 4))
        x = rng.random(4)
        before = np.linalg.norm(w - x, axis=1)
        update_weights(w, pos, x, 14, 0.4, 2.0)
        after = np.linalg.norm(w - x, axis=1)
        self.assertTrue(np.all(after <= before))
        shrink = after / before                    # (1 - eta * h) for every neuron
        self.assertAlmostEqual(shrink[14], 0.6)
        self.assertEqual(shrink.argmin(), 14)

    def test_full_rate_tiny_radius_copies_input(self):
        w = np.zeros((9, 3))
        update_weights(w, grid_positions(3, 3), [0.2, 0.4, 0.6], 4, 1.0, 1e-3)
        np.testing.assert_allclose(w[4], [0.2, 0.4, 0.6])
        np.testing.assert_allclose(np.delete(w, 4, axis=0), 0.0)

    def test_train_step(self):
        som = SOM(2, 2, 2, seed=0)
        som.weights = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        j, d = som.train_step([0.9, 0.9], 0.5, 0.5)
        self.assertEqual(j, 3)
        self.assertAlmostEqual(d, math.hypot(0.1, 0.1))
        np.testing.assert_allclose(som.weights[3], [0.95, 0.95])


def train(som, x, params):
    """Small training loop used only by these tests (the real one comes later)."""
    for epoch in range(params["maxIterations"]):
        eta, sigma = schedule(params, epoch)
        for i in som.rng.permutation(len(x)):
            som.train_step(x[i], eta, sigma)


class TestSyntheticOrdering(unittest.TestCase):
    def test_chain_orders_itself_on_a_line(self):
        # A 1 x 10 chain trained on numbers spread over [0, 1] must end up
        # sorted (increasing or decreasing) and cover the range: the classic
        # self-organization result.
        params = {"maxIterations": 30, "initialLearningRate": 0.5, "finalLearningRate": 0.02,
                  "initialRadius": 5.0, "finalRadius": 0.5}
        x = np.random.default_rng(0).random((300, 1))
        som = SOM(1, 10, 1, seed=1)
        train(som, x, params)
        w = som.weights[:, 0]
        steps = np.diff(w)
        self.assertTrue(np.all(steps > 0) or np.all(steps < 0), w)
        self.assertLess(min(w), 0.15)
        self.assertGreater(max(w), 0.85)

    def test_four_clusters_go_to_the_corners(self):
        # Four tight groups at the corners of the unit square: after training,
        # each group's winners sit in a different corner region of a 6x6 map
        # and the quantization error is small.
        rng = np.random.default_rng(2)
        centres = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])
        x = np.vstack([c + 0.02 * rng.standard_normal((50, 2)) for c in centres])
        labels = np.repeat(np.arange(4), 50)
        params = {"maxIterations": 20, "initialLearningRate": 0.5, "finalLearningRate": 0.01,
                  "initialRadius": 3.0, "finalRadius": 0.5}
        som = SOM(6, 6, 2, seed=3)
        train(som, x, params)
        idx, dist = som.winners(x)
        self.assertLess(dist.mean(), 0.05)
        groups = [set(idx[labels == k]) for k in range(4)]
        for a in range(4):
            for b in range(a + 1, 4):
                self.assertFalse(groups[a] & groups[b])      # no neuron shared by two groups


class TestRealData(unittest.TestCase):
    """One pass over (part of) the letter data with the sample parameters."""

    @classmethod
    def setUpClass(cls):
        cls.params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
        cls.x = load_train_test(cls.params, CODE_DIR).x_train[:2000]

    def one_pass(self, init_method, epoch):
        """Quantization error before and after one pass at `epoch`'s eta and radius."""
        som = SOM.from_parameters(dict(self.params, initMethod=init_method), self.x)
        before = som.winners(self.x)[1].mean()
        eta, sigma = schedule(self.params, epoch)
        for i in som.rng.permutation(len(self.x)):
            som.train_step(self.x[i], eta, sigma)
        return before, som.winners(self.x)[1].mean()

    def test_first_pass_improves_random_start(self):
        before, after = self.one_pass("random", 0)
        self.assertLess(after, before)

    def test_wide_radius_pulls_the_map_together(self):
        # With the initial radius (half the grid) every neuron follows almost
        # every pattern, so the map first contracts towards the data mean and
        # the error of a data-sample start goes UP in the first epoch. This is
        # the expected ordering phase, not a bug.
        before, after = self.one_pass("sample", 0)
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
