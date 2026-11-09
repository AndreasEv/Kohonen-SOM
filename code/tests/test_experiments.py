"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Tests for the experiment runner (experiments.py): reading experiments.txt,
merging each line with parameters.txt, sharing runs with equal settings,
re-using finished runs, the seed statistics and the summary tables. Training
runs use the tiny 4-cluster letter data set of test_clustering.py, so the
whole file takes a few seconds.
"""

import os
import tempfile
import unittest

import numpy as np

import experiments as ex
import main
from parameters import parse_parameters
from tests.test_clustering import write_letter_files

BASE_LINES = ["gridRows 20", "gridCols 20", "numInputs 16", "maxIterations 100",
              "initialLearningRate 0.5", "finalLearningRate 0.01", "initialRadius 10",
              "finalRadius 1", "randomSeed 42", "initMethod sample",
              "trainFile training.txt", "testFile test.txt"]
BASE = parse_parameters(BASE_LINES)


def parse(text):
    return ex.parse_experiments(text.splitlines(), BASE)


def fake_result(train, test, acc):
    return ex.RunResult(initial_train_error=0.3, final_train_error=train, final_test_error=test,
                        accuracy=acc, correct=int(acc * 100), total=100, empty_neurons=2,
                        seconds=1.0)


class ParseExperimentsTest(unittest.TestCase):
    def test_overrides_change_only_named_parameters(self):
        exps = parse("# comment\n\ngrid small gridRows 10 gridCols 10  # trailing\nbase b\n")
        self.assertEqual([e.name for e in exps], ["small", "b"])
        small = exps[0]
        self.assertEqual(small.group, "grid")
        self.assertEqual(small.overrides, {"gridRows": "10", "gridCols": "10"})
        self.assertEqual((small.params["gridRows"], small.params["gridCols"]), (10, 10))
        for name in BASE:
            if name not in ("gridRows", "gridCols"):
                self.assertEqual(small.params[name], BASE[name])
        self.assertEqual(exps[1].params, BASE)

    def test_values_are_typed_and_validated(self):
        exp = parse("g a initialLearningRate 0.1 initMethod random randomSeed 3")[0]
        self.assertIsInstance(exp.params["initialLearningRate"], float)
        self.assertEqual(exp.params["randomSeed"], 3)
        self.assertEqual(exp.params["initMethod"], "random")

    def test_bad_lines(self):
        bad = {
            "g": "expected",                                   # no name
            "g a\ng a": "used twice",                          # repeated name
            "g a gridSize 3": "unknown parameter",
            "g a gridRows": "no value",
            "g a gridRows 3 gridRows 4": "given twice",
            "g a gridRows 0": "gridRows must be",              # fails the parameter check
            "g a finalRadius 20": "finalRadius",               # final > initial radius
            "g a initMethod pca": "initMethod",
            "# only a comment": "no experiments",
        }
        for text, message in bad.items():
            with self.subTest(text=text):
                with self.assertRaises(ex.ExperimentError) as caught:
                    parse(text)
                self.assertIn(message, str(caught.exception))

    def test_error_names_the_line(self):
        with self.assertRaises(ex.ExperimentError) as caught:
            parse("g a\n\ng b gridRows x\n")
        self.assertIn("line 3", str(caught.exception))

    def test_shipped_file_is_valid(self):
        # experiments.txt in code/ must parse against the shipped parameters.txt.
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = ex.read_parameters(os.path.join(here, "parameters.txt"))
        exps = ex.read_experiments(os.path.join(here, "experiments.txt"), base)
        groups = {e.group for e in exps}
        self.assertTrue({"grid", "learning", "radius", "init", "seeds"} <= groups)
        grids = {(e.params["gridRows"], e.params["gridCols"]) for e in exps if e.group == "grid"}
        self.assertGreaterEqual(len(grids), 2)
        seeds = [e for e in exps if e.group == "seeds"]
        self.assertGreaterEqual(len(seeds), 3)
        self.assertTrue(ex.is_seed_group(seeds))


class PlanRunsTest(unittest.TestCase):
    def test_equal_settings_share_the_first_run(self):
        exps = parse("base b\ngrid g20\ngrid g10 gridRows 10\nseeds s42 randomSeed 42\n"
                     "seeds s1 randomSeed 1\n")
        self.assertEqual(ex.plan_runs(exps),
                         {"b": "b", "g20": "b", "g10": "g10", "s42": "b", "s1": "s1"})

    def test_seed_group_detection(self):
        seeds = parse("s a randomSeed 1\ns b randomSeed 2\ns c randomSeed 3")
        self.assertTrue(ex.is_seed_group(seeds))
        self.assertFalse(ex.is_seed_group(seeds[:1]))
        self.assertFalse(ex.is_seed_group(parse("g a\ng b gridRows 10")))
        self.assertFalse(ex.is_seed_group(parse("s a randomSeed 1\ns b randomSeed 2 gridRows 3")))
        self.assertFalse(ex.is_seed_group(parse("s a randomSeed 1\ns b randomSeed 1")))


class FormattingTest(unittest.TestCase):
    def test_parameters_text_round_trips(self):
        exp = parse("g a gridRows 7 finalRadius 0.5")[0]
        text = ex.format_parameters(exp.params)
        self.assertEqual(parse_parameters(text.splitlines()), exp.params)

    def test_run_text_round_trips(self):
        r = fake_result(0.281234, 0.284321, 0.558)
        back = ex.parse_run(ex.format_run(r))
        self.assertEqual(back.correct, r.correct)
        self.assertEqual(back.empty_neurons, 2)
        self.assertAlmostEqual(back.final_test_error, r.final_test_error, places=6)
        with self.assertRaises((ValueError, KeyError)):
            ex.parse_run("accuracy 0.5\n")

    def test_seed_statistics(self):
        stats = ex.seed_statistics([fake_result(0.2, 0.3, 0.5), fake_result(0.4, 0.5, 0.7),
                                    fake_result(0.3, 0.4, 0.6)])
        mean, std = stats["final_train_error"]
        self.assertAlmostEqual(mean, 0.3)
        self.assertAlmostEqual(std, 0.1)            # sample std (n-1) of 0.2, 0.3, 0.4
        self.assertAlmostEqual(stats["accuracy"][0], 0.6)

    def test_summary_tables(self):
        exps = parse("base b\ngrid g10 gridRows 10 gridCols 10\ngrid g20\n"
                     "seeds s42\nseeds s1 randomSeed 1\nseeds s2 randomSeed 2\n")
        res = {"b": fake_result(0.28, 0.29, 0.55), "g10": fake_result(0.35, 0.36, 0.45),
               "g20": fake_result(0.28, 0.29, 0.55), "s42": fake_result(0.28, 0.29, 0.55),
               "s1": fake_result(0.30, 0.31, 0.53), "s2": fake_result(0.26, 0.27, 0.57)}
        text = "\n".join(ex.format_summary(exps, res))
        self.assertEqual([l for l in text.splitlines() if l.startswith("Group:")],
                         ["Group: base", "Group: grid", "Group: seeds"])
        self.assertIn("g20 (= b)", text)
        self.assertIn("10x10", text)
        self.assertIn("0.5->0.01", text)
        self.assertIn("10->1", text)
        self.assertIn("45.00", text)
        # Only the seed group gets mean +- std: (0.28, 0.30, 0.26) -> 0.28 +- 0.02.
        stats = [l for l in text.splitlines() if l.startswith("Mean")]
        self.assertEqual(len(stats), 1)
        self.assertIn("over 3 seeds: train QE 0.2800 +- 0.0200", stats[0])
        self.assertIn("accuracy 55.00 +- 2.00 %", stats[0])

        csv = ex.format_csv(exps, res)
        self.assertEqual(len(csv), 1 + len(exps))
        header = csv[0].split(",")
        row = dict(zip(header, csv[3].split(",")))       # g20
        self.assertEqual(row["same_as"], "b")
        self.assertEqual(row["gridRows"], "20")
        self.assertEqual(row["initialRadius"], "10")
        self.assertEqual(row["accuracy"], "0.550000")
        self.assertTrue(all(len(line.split(",")) == len(header) for line in csv))


class RunTest(unittest.TestCase):
    """Real training runs on the tiny 4-letter data set (4x4 map, 8 epochs)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        write_letter_files(self.folder)
        self.write_experiments("base b\n"
                               "grid g3 gridRows 3 gridCols 3\n"
                               "grid g4\n"
                               "seeds s7\n"
                               "seeds s1 randomSeed 1\n"
                               "seeds s2 randomSeed 2\n")
        self.lines = []

    def tearDown(self):
        self.tmp.cleanup()

    def write_experiments(self, text):
        with open(os.path.join(self.folder, "experiments.txt"), "w") as f:
            f.write(text)

    def run_exp(self, **kwargs):
        return ex.run(self.folder, log=self.lines.append, **kwargs)

    def out(self, *parts):
        return os.path.join(self.folder, "experiments", *parts)

    def test_runs_write_files_and_summary(self):
        exps, results = self.run_exp()
        self.assertEqual(set(results), {e.name for e in exps})
        # One folder per distinct setting: b, g3, s1, s2 (g4 and s7 equal b).
        folders = sorted(d for d in os.listdir(self.out()) if os.path.isdir(self.out(d)))
        self.assertEqual(folders, ["b", "g3", "s1", "s2"])
        for name in folders:
            for f in ("parameters.txt", "results.txt", "clustering.txt", "run.txt"):
                self.assertTrue(os.path.isfile(self.out(name, f)), (name, f))
        with open(self.out("g3", "clustering.txt"), encoding="utf-8") as f:
            self.assertEqual(len(f.read().splitlines()), 9)
        with open(self.out("b", "results.txt")) as f:
            self.assertEqual(len(f.read().splitlines()), 8)
        self.assertIs(results["g4"], results["b"])
        self.assertIs(results["s7"], results["b"])
        self.assertEqual(results["b"].total, 20)
        self.assertGreaterEqual(results["b"].accuracy, 0.95)
        self.assertLess(results["b"].final_train_error, 0.1)   # clusters are 0.03 wide
        self.assertGreater(results["b"].initial_train_error, 0.0)
        with open(self.out("summary.txt"), encoding="utf-8") as f:
            summary = f.read()
        self.assertIn("Group: seeds", summary)
        self.assertIn("Mean +- std over 3 seeds", summary)
        with open(self.out("summary.csv"), encoding="utf-8") as f:
            self.assertEqual(len(f.read().splitlines()), 1 + 6)

    def test_same_result_as_main_program(self):
        # The baseline run must give exactly what main.py gives with parameters.txt.
        _, results = self.run_exp(only="base")
        _, _, history, _, acc = main.run(self.folder, verbose=False)
        with open(os.path.join(self.folder, "results.txt")) as a, open(self.out("b", "results.txt")) as b:
            self.assertEqual(a.read(), b.read())
        with open(os.path.join(self.folder, "clustering.txt"), encoding="utf-8") as a, \
                open(self.out("b", "clustering.txt"), encoding="utf-8") as b:
            self.assertEqual(a.read(), b.read())
        self.assertEqual(results["b"].final_train_error, history[-1].train_error)
        self.assertEqual(results["b"].correct, acc.correct)

    def test_finished_runs_are_loaded_not_retrained(self):
        _, first = self.run_exp()
        stamp = os.path.getmtime(self.out("b", "run.txt"))
        self.lines.clear()
        _, second = self.run_exp()
        self.assertEqual(os.path.getmtime(self.out("b", "run.txt")), stamp)
        self.assertFalse(any("Training" in line for line in self.lines))
        for name in first:
            self.assertAlmostEqual(first[name].final_test_error, second[name].final_test_error, 6)
        # A changed setting under the same name is retrained.
        self.write_experiments("base b\ngrid g3 gridRows 3 gridCols 2\n")
        self.lines.clear()
        _, third = self.run_exp()
        self.assertIn("Training 1 run(s) ...", self.lines)
        with open(self.out("g3", "clustering.txt"), encoding="utf-8") as f:
            self.assertEqual(len(f.read().splitlines()), 6)
        # --force retrains everything.
        self.lines.clear()
        self.run_exp(force=True)
        self.assertIn("Training 2 run(s) ...", self.lines)

    def test_only_one_group(self):
        _, results = self.run_exp(only="grid")
        self.assertEqual(set(results), {"b", "g3", "g4", "s7"})   # s7 equals b, so it is known
        self.assertFalse(os.path.exists(self.out("s1")))
        with open(self.out("summary.txt"), encoding="utf-8") as f:
            summary = f.read()
        self.assertIn("Group: grid", summary)
        self.assertNotIn("s1", summary)
        with self.assertRaises(ex.ExperimentError):
            self.run_exp(only="nothing")

    def test_parallel_gives_the_same_numbers(self):
        _, serial = self.run_exp(out="serial")
        _, parallel = self.run_exp(out="parallel", jobs=2)
        for name in serial:
            self.assertEqual(serial[name].final_train_error, parallel[name].final_train_error)
            self.assertEqual(serial[name].correct, parallel[name].correct)
        with open(os.path.join(self.folder, "serial", "s2", "results.txt")) as a, \
                open(os.path.join(self.folder, "parallel", "s2", "results.txt")) as b:
            self.assertEqual(a.read(), b.read())

    def test_seeds_give_different_maps(self):
        _, results = self.run_exp(only="seeds")
        errors = {results[n].final_train_error for n in ("s7", "s1", "s2")}
        self.assertEqual(len(errors), 3)
        w = [np.loadtxt(self.out(n, "results.txt")) for n in ("b", "s1")]
        self.assertFalse(np.array_equal(w[0], w[1]))

    def test_command_line(self):
        cwd = os.getcwd()
        os.chdir(self.folder)
        try:
            import contextlib
            import io
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = ex.main(["--only", "grid"])
            self.assertEqual(code, 0)
            self.assertIn("Group: grid", buffer.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ex.main(["--file", "missing.txt"]), 1)
            self.assertIn("missing.txt", err.getvalue())
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
