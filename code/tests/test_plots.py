"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Tests for plots.py: reading results.txt / clustering.txt, drawing every plot
to a PNG file without a screen (Agg backend), and the plots main.py saves.
Drawing tests are skipped when matplotlib is not installed.
"""

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import plots
from clustering import NO_LABEL, Accuracy

try:
    import matplotlib  # noqa: F401
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

needs_matplotlib = unittest.skipUnless(HAVE_MATPLOTLIB, "matplotlib is not installed")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

RESULTS = "1 0.512722 0.510981\n2 0.549014 0.548287\n3 0.431693 0.437507\n"
# 2x3 map: one empty neuron, one impure neuron
CLUSTERING = ("0 0 A 10 10\n0 1 A 4 3\n0 2 B 6 6\n"
              f"1 0 {NO_LABEL} 0 0\n1 1 C 8 5\n1 2 B 2 2\n")


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def is_png(path):
    with open(path, "rb") as f:
        return f.read(8) == PNG_SIGNATURE


class ReadResultsTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.folder.name, "results.txt")

    def tearDown(self):
        self.folder.cleanup()

    def test_columns(self):
        write(self.path, RESULTS)
        epochs, train, test = plots.read_results(self.path)
        np.testing.assert_array_equal(epochs, [1, 2, 3])
        np.testing.assert_allclose(train, [0.512722, 0.549014, 0.431693])
        np.testing.assert_allclose(test, [0.510981, 0.548287, 0.437507])

    def test_single_epoch(self):
        write(self.path, "1 0.5 0.6\n")
        epochs, train, test = plots.read_results(self.path)
        self.assertEqual((len(epochs), train[0], test[0]), (1, 0.5, 0.6))

    def test_missing_empty_or_wrong_columns(self):
        with self.assertRaises(plots.PlotError):
            plots.read_results(self.path)
        for text in ["", "1 0.5\n2 0.4\n", "1 abc 0.5\n"]:
            write(self.path, text)
            with self.assertRaises(plots.PlotError, msg=repr(text)):
                plots.read_results(self.path)


class ReadClusteringTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.folder.name, "clustering.txt")

    def tearDown(self):
        self.folder.cleanup()

    def test_fields_and_shape(self):
        write(self.path, CLUSTERING)
        c = plots.read_clustering(self.path)
        self.assertEqual(c["shape"], (2, 3))
        self.assertEqual(list(c["letters"]), ["A", "A", "B", NO_LABEL, "C", "B"])
        np.testing.assert_array_equal(c["patterns"], [10, 4, 6, 0, 8, 2])
        np.testing.assert_array_equal(c["hits"], [10, 3, 6, 0, 5, 2])

    def test_bad_files(self):
        with self.assertRaises(plots.PlotError):
            plots.read_clustering(self.path)
        for text in ["", "0 0 A 10\n", "0 x A 10 10\n", "-1 0 A 1 1\n"]:
            write(self.path, text)
            with self.assertRaises(plots.PlotError, msg=repr(text)):
                plots.read_clustering(self.path)

    def test_reads_what_results_py_writes(self):
        """The reader matches the real writer, including the "–" label."""
        from clustering import label_neurons
        from results import write_clustering
        from som import SOM

        som = SOM(2, 2, 3, seed=0)
        labelling = label_neurons(np.array([0, 0, 1, 3]), np.array(["A", "B", "B", "C"]), 4)
        write_clustering(self.path, som, labelling)
        c = plots.read_clustering(self.path)
        self.assertEqual(c["shape"], (2, 2))
        self.assertEqual(list(c["letters"]), ["A", "B", NO_LABEL, "C"])
        np.testing.assert_array_equal(c["patterns"], [2, 1, 0, 1])


@needs_matplotlib
class DrawingTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.folder.name, "plots")   # does not exist yet

    def tearDown(self):
        self.folder.cleanup()

    def test_backend_is_headless(self):
        plots._pyplot()
        self.assertEqual(matplotlib.get_backend().lower(), "agg")

    def test_error_curve(self):
        path = plots.plot_error_curve([1, 2, 3], [0.5, 0.4, 0.3], [0.52, 0.41, 0.31],
                                      os.path.join(self.out, "e.png"))
        self.assertTrue(is_png(path))

    def test_error_curve_one_epoch_and_bad_lengths(self):
        self.assertTrue(is_png(plots.plot_error_curve([1], [0.5], [0.6],
                                                      os.path.join(self.out, "e1.png"))))
        with self.assertRaises(ValueError):
            plots.plot_error_curve([1, 2], [0.5], [0.6, 0.7], os.path.join(self.out, "bad.png"))

    def test_letter_map(self):
        clustering_path = os.path.join(self.folder.name, "clustering.txt")
        write(clustering_path, CLUSTERING)
        path = plots.plot_letter_map(plots.read_clustering(clustering_path),
                                     os.path.join(self.out, "m.png"))
        self.assertTrue(is_png(path))

    def test_letter_map_large_grid(self):
        lines = [f"{r} {c} {chr(65 + (r * 30 + c) % 26)} 3 2" for r in range(30) for c in range(30)]
        clustering_path = os.path.join(self.folder.name, "clustering.txt")
        write(clustering_path, "\n".join(lines) + "\n")
        self.assertTrue(is_png(plots.plot_letter_map(plots.read_clustering(clustering_path),
                                                     os.path.join(self.out, "big.png"))))

    def test_letter_accuracy(self):
        per_letter = {"A": (9, 10), "B": (1, 4), "C": (0, 3)}
        path = plots.plot_letter_accuracy(per_letter, os.path.join(self.out, "a.png"), overall=10 / 17)
        self.assertTrue(is_png(path))
        with self.assertRaises(ValueError):
            plots.plot_letter_accuracy({}, os.path.join(self.out, "none.png"))

    def test_no_figures_left_open(self):
        plt = plots._pyplot()
        plots.plot_error_curve([1, 2], [0.5, 0.4], [0.5, 0.4], os.path.join(self.out, "e.png"))
        plots.plot_letter_accuracy({"A": (1, 2)}, os.path.join(self.out, "a.png"))
        self.assertEqual(plt.get_fignums(), [])


@needs_matplotlib
class SavePlotsTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        write(os.path.join(self.folder.name, "results.txt"), RESULTS)
        write(os.path.join(self.folder.name, "clustering.txt"), CLUSTERING)

    def tearDown(self):
        self.folder.cleanup()

    def plot_path(self, name):
        return os.path.join(self.folder.name, plots.PLOTS_FOLDER, name)

    def test_all_three_with_accuracy(self):
        acc = Accuracy(correct=10, total=17, rate=10 / 17,
                       per_letter={"A": (9, 10), "B": (1, 4), "C": (0, 3)})
        written = plots.save_plots(self.folder.name, acc)
        expected = [self.plot_path(n) for n in (plots.ERROR_PLOT, plots.MAP_PLOT, plots.ACCURACY_PLOT)]
        self.assertEqual(written, expected)
        for path in expected:
            self.assertTrue(is_png(path), path)

    def test_without_accuracy_skips_bar_chart(self):
        written = plots.save_plots(self.folder.name)
        self.assertEqual(len(written), 2)
        self.assertFalse(os.path.exists(self.plot_path(plots.ACCURACY_PLOT)))

    def test_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(plots.main([self.folder.name]), 0)
        self.assertTrue(is_png(self.plot_path(plots.ERROR_PLOT)))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(plots.main([os.path.join(self.folder.name, "missing")]), 1)


class MainPlotsTest(unittest.TestCase):
    """main.py saves the plots after a run and survives a missing matplotlib."""

    def test_missing_matplotlib_is_not_an_error(self):
        import main
        acc = Accuracy(correct=1, total=1, rate=1.0, per_letter={"A": (1, 1)})
        with mock.patch.object(plots, "save_plots", side_effect=ImportError("no matplotlib")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main.print_plots(acc), [])
        self.assertIn("matplotlib is not installed", out.getvalue())

    @needs_matplotlib
    def test_saves_plots_of_a_run(self):
        import main
        with tempfile.TemporaryDirectory() as folder:
            write(os.path.join(folder, "results.txt"), RESULTS)
            write(os.path.join(folder, "clustering.txt"), CLUSTERING)
            acc = Accuracy(correct=1, total=2, rate=0.5, per_letter={"A": (1, 1), "B": (0, 1)})
            with contextlib.redirect_stdout(io.StringIO()):
                written = main.print_plots(acc, folder)
            self.assertEqual(len(written), 3)
            self.assertTrue(all(is_png(p) for p in written))


if __name__ == "__main__":
    unittest.main()
