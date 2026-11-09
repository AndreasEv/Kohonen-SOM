"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Tests for the neuron labels and the clustering accuracy (clustering.py),
the clustering.txt writer (results.py) and the full program (main.run).
Small hand-made examples check the majority vote, the tie rule and the "–"
mark; the synthetic 4-cluster data set checks that a trained map labels and
classifies known clusters perfectly.
"""

import os
import tempfile
import unittest

import numpy as np

import main
from clustering import NO_LABEL, accuracy, classify, evaluate, label_neurons
from results import format_clustering, write_clustering
from som import SOM
from training import train
from tests.test_training import LETTERS, PARAMS, make_clusters


class TestLabelNeurons(unittest.TestCase):
    def test_majority_letter(self):
        # Neuron 0 wins A, A, B -> A. Neuron 2 wins C -> C. Neuron 1 wins nothing.
        lab = label_neurons([0, 0, 0, 2], ["A", "A", "B", "C"], 3)
        self.assertEqual(list(lab.labels), ["A", NO_LABEL, "C"])
        self.assertEqual(list(lab.patterns), [3, 0, 1])
        self.assertEqual(list(lab.hits), [2, 0, 1])
        self.assertEqual(lab.letters, ["A", "B", "C"])
        np.testing.assert_array_equal(lab.counts, [[2, 1, 0], [0, 0, 0], [0, 0, 1]])

    def test_tie_goes_to_first_letter_whatever_the_order(self):
        for labels in (["B", "A", "B", "A"], ["A", "B", "A", "B"], ["B", "B", "A", "A"]):
            self.assertEqual(label_neurons([0, 0, 0, 0], labels, 1).labels[0], "A")

    def test_neuron_without_patterns_gets_dash(self):
        lab = label_neurons([1], ["Z"], 4)
        self.assertEqual(list(lab.labels), [NO_LABEL, "Z", NO_LABEL, NO_LABEL])
        self.assertEqual(NO_LABEL, "–")

    def test_patterns_add_up(self):
        rng = np.random.default_rng(0)
        winners = rng.integers(0, 9, 200)
        labels = rng.choice(list("ABCDE"), 200)
        lab = label_neurons(winners, labels, 9)
        self.assertEqual(lab.patterns.sum(), 200)
        self.assertTrue(np.all(lab.hits <= lab.patterns))
        for j in range(9):
            mine = labels[winners == j]
            if len(mine):
                top = max(np.sum(mine == l) for l in "ABCDE")
                self.assertEqual(lab.hits[j], top)
                self.assertEqual(np.sum(mine == lab.labels[j]), top)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            label_neurons([0, 1], ["A"], 2)
        with self.assertRaises(ValueError):
            label_neurons([5], ["A"], 2)


class TestAccuracy(unittest.TestCase):
    def test_classify_uses_winner_label(self):
        labels = np.array(["A", NO_LABEL, "C"], dtype=object)
        self.assertEqual(list(classify([2, 0, 1, 2], labels)), ["C", "A", NO_LABEL, "C"])

    def test_overall_and_per_letter(self):
        acc = accuracy(["A", "B", NO_LABEL, "A", "C"], ["A", "A", "A", "B", "C"])
        self.assertEqual((acc.correct, acc.total), (2, 5))
        self.assertAlmostEqual(acc.rate, 0.4)
        self.assertEqual(acc.per_letter, {"A": (1, 3), "B": (0, 1), "C": (1, 1)})

    def test_unlabelled_winner_is_wrong(self):
        self.assertEqual(accuracy([NO_LABEL], ["A"]).correct, 0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            accuracy(["A"], ["A", "B"])


class TestTrainedMap(unittest.TestCase):
    """Train on 4 clusters lettered A-D: labels and test accuracy must be perfect."""

    @classmethod
    def setUpClass(cls):
        x_train, y_train = make_clusters(40, seed=10)
        x_test, y_test = make_clusters(10, seed=11)
        cls.letters_train = np.array([LETTERS[k] for k in y_train])
        cls.letters_test = np.array([LETTERS[k] for k in y_test])
        cls.x_train, cls.x_test = x_train, x_test
        cls.som = SOM(5, 5, 2, seed=3, init_method="random")
        train(cls.som, x_train, x_test, PARAMS)
        cls.weights = cls.som.weights.copy()
        cls.labelling, cls.acc = evaluate(cls.som, x_train, cls.letters_train,
                                          x_test, cls.letters_test)

    def test_every_letter_has_neurons_and_they_are_pure(self):
        used = [l for l in self.labelling.labels if l != NO_LABEL]
        self.assertEqual(set(used), set(LETTERS))
        np.testing.assert_array_equal(self.labelling.hits, self.labelling.patterns)

    def test_perfect_test_accuracy(self):
        self.assertEqual(self.acc.correct, self.acc.total)
        self.assertEqual(self.acc.per_letter, {l: (10, 10) for l in LETTERS})

    def test_evaluate_does_not_change_weights(self):
        np.testing.assert_array_equal(self.som.weights, self.weights)

    def test_clustering_lines(self):
        lines = format_clustering(self.som, self.labelling)
        self.assertEqual(len(lines), 25)
        for j, line in enumerate(lines):
            row, col, letter, patterns, hits = line.split(" ")
            self.assertEqual((int(row), int(col)), self.som.position(j))
            self.assertEqual(letter, self.labelling.labels[j])
            self.assertEqual(int(patterns), self.labelling.patterns[j])
            self.assertEqual(int(hits), self.labelling.hits[j])

    def test_clustering_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "clustering.txt")
            write_clustering(path, self.som, self.labelling)
            with open(path, encoding="utf-8") as f:
                rows = [line.split() for line in f]
        self.assertEqual([(int(r[0]), int(r[1])) for r in rows],
                         [(r, c) for r in range(5) for c in range(5)])
        self.assertEqual(sum(int(r[3]) for r in rows), len(self.x_train))


def write_letter_files(folder, train_letters=None):
    """Tiny data folder for main.run (4 clusters, letters A-D). Returns nothing."""
    x_train, y_train = make_clusters(20, seed=20)
    x_test, y_test = make_clusters(5, seed=21)
    if train_letters is None:
        train_letters = [LETTERS[k] for k in y_train]
    for name, x, letters in (("tr.txt", x_train, train_letters),
                             ("te.txt", x_test, [LETTERS[k] for k in y_test])):
        with open(os.path.join(folder, name), "w") as f:
            for row, letter in zip(np.round(x * 15).astype(int), letters):
                f.write(letter + "," + ",".join(map(str, row)) + "\n")
    with open(os.path.join(folder, "parameters.txt"), "w") as f:
        f.write("gridRows 4\ngridCols 4\nnumInputs 2\nmaxIterations 8\n"
                "initialLearningRate 0.5\nfinalLearningRate 0.05\n"
                "initialRadius 2\nfinalRadius 0.5\nrandomSeed 7\n"
                "initMethod sample\ntrainFile tr.txt\ntestFile te.txt\n")
    return len(y_train)


class TestProgram(unittest.TestCase):
    def test_run_writes_clustering_and_scores_test_set(self):
        with tempfile.TemporaryDirectory() as folder:
            write_letter_files(folder)
            som, _, _, labelling, acc = main.run(folder, verbose=False)
            with open(os.path.join(folder, "clustering.txt"), encoding="utf-8") as f:
                lines = f.read().splitlines()
        self.assertEqual(lines, format_clustering(som, labelling))
        self.assertEqual(len(lines), 16)
        self.assertEqual(acc.total, 20)
        self.assertGreaterEqual(acc.rate, 0.95)

    def test_letters_do_not_affect_training(self):
        # Same features, scrambled training letters: the weights and
        # results.txt must be identical; only the labels may differ.
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            n = write_letter_files(a)
            write_letter_files(b, train_letters=list(np.random.default_rng(1).choice(list("WXYZ"), n)))
            som_a, *_ = main.run(a, verbose=False)
            som_b, *_ = main.run(b, verbose=False)
            with open(os.path.join(a, "results.txt")) as fa, open(os.path.join(b, "results.txt")) as fb:
                self.assertEqual(fa.read(), fb.read())
        np.testing.assert_array_equal(som_a.weights, som_b.weights)


if __name__ == "__main__":
    unittest.main()
