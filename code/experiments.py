"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Experiment runner for the report: trains the SOM once per line of
experiments.txt and collects the results in summary tables.

Every experiment starts from parameters.txt in the current folder (the
baseline) and changes only the parameters named on its line, e.g.
    grid  grid_10x10  gridRows 10 gridCols 10
so the comparisons (grid size, learning rate, radius, initialization) and the
repeated runs with other seeds all differ from the baseline in one thing.
The SOM itself is not changed: every run goes through the same code as
main.py (SOM, train, evaluate).

Output, in experiments/ (or the folder given with --out):
  <name>/parameters.txt   the full settings of the run
  <name>/results.txt      training / testing error per epoch (as main.py)
  <name>/clustering.txt   neuron letters (as main.py)
  <name>/run.txt          final numbers of the run (read back on a re-run)
  summary.txt             one table per group; a group whose runs differ only
                          in randomSeed also gets the mean and standard
                          deviation (sample std, n-1) of every result
  summary.csv             the same numbers, one row per experiment line

A finished run is not trained again when its folder already holds the same
parameters.txt and a run.txt (use --force to retrain), so adding a line to
experiments.txt only trains the new run. Lines with exactly the same
settings are trained once and share the result.

Usage (from the code/ folder):
    python3 experiments.py                  all experiments, one at a time
    python3 experiments.py --jobs 4         four runs in parallel
    python3 experiments.py --only seeds     train only the lines of one group
"""

import argparse
import os
import sys
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from clustering import NO_LABEL, evaluate
from data import DataError, load_train_test
from parameters import SCHEMA, ParameterError, parse_parameters, read_parameters
from results import CLUSTERING_FILE, RESULTS_FILE, write_clustering, write_results
from som import SOM
from training import train

PARAMETERS_FILE = "parameters.txt"
EXPERIMENTS_FILE = "experiments.txt"
OUTPUT_FOLDER = "experiments"
RUN_FILE = "run.txt"
SUMMARY_FILE = "summary.txt"
SUMMARY_CSV = "summary.csv"


class ExperimentError(ValueError):
    """Raised when experiments.txt has a bad line."""


# One line of experiments.txt, with the merged and validated settings.
#   overrides  {name: text} as written on the line (empty for the baseline)
#   params     the full parameter dict (parameters.txt + overrides)
Experiment = namedtuple("Experiment", ["group", "name", "overrides", "params"])

# The final numbers of one training run (also stored in run.txt).
RunResult = namedtuple("RunResult", [
    "initial_train_error", "final_train_error", "final_test_error",
    "accuracy", "correct", "total", "empty_neurons", "seconds"])

_INT_FIELDS = ("correct", "total", "empty_neurons")


# ---------------------------------------------------------------- reading

def merge_parameters(base, overrides, source="experiments.txt"):
    """Baseline parameters with `overrides` ({name: text}) applied, validated.

    The merged settings go through the same checks as parameters.txt
    (parameters.parse_parameters), so e.g. finalRadius > initialRadius is
    rejected here too. Raises ParameterError.
    """
    lines = [f"{name} {overrides.get(name, base[name])}" for name in SCHEMA]
    return parse_parameters(lines, source=source)


def parse_experiments(lines, base, source=EXPERIMENTS_FILE):
    """Parse the lines of experiments.txt into a list of Experiment.

    Blank lines and '#' comments are ignored. Each line is
    "group name [parameter value ...]". Raises ExperimentError naming the
    line for a missing name, a repeated name, an unknown or repeated
    parameter, a missing value or an invalid merged setting.
    """
    experiments, names = [], set()
    for number, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{source}, line {number}"
        parts = line.split()
        if len(parts) < 2:
            raise ExperimentError(f"{where}: expected 'group name [parameter value ...]'")
        group, name, rest = parts[0], parts[1], parts[2:]
        if name in names:
            raise ExperimentError(f"{where}: experiment name {name!r} is used twice")
        if os.sep in name or name in (".", ".."):
            raise ExperimentError(f"{where}: {name!r} cannot be used as a folder name")
        if len(rest) % 2:
            raise ExperimentError(f"{where}: parameter {rest[-1]!r} has no value")
        overrides = {}
        for key, value in zip(rest[0::2], rest[1::2]):
            if key not in SCHEMA:
                raise ExperimentError(f"{where}: unknown parameter {key!r}")
            if key in overrides:
                raise ExperimentError(f"{where}: {key} is given twice")
            overrides[key] = value
        try:
            params = merge_parameters(base, overrides, source=where)
        except ParameterError as error:
            raise ExperimentError(str(error)) from None
        names.add(name)
        experiments.append(Experiment(group, name, overrides, params))
    if not experiments:
        raise ExperimentError(f"{source}: no experiments")
    return experiments


def read_experiments(path, base):
    """Read and validate experiments.txt at `path` (see parse_experiments)."""
    if not os.path.isfile(path):
        raise ExperimentError(f"{path}: file not found")
    with open(path, encoding="utf-8") as f:
        return parse_experiments(f, base, source=os.path.basename(path))


# ---------------------------------------------------------------- running

def format_parameters(params):
    """Text of a parameters.txt holding `params` (same order as SCHEMA)."""
    return "".join(f"{name} {params[name]}\n" for name in SCHEMA)


def format_run(result):
    """Text of run.txt for a RunResult."""
    lines = []
    for field, value in result._asdict().items():
        lines.append(f"{field} {value}" if field in _INT_FIELDS else f"{field} {value:.6f}")
    return "\n".join(lines) + "\n"


def parse_run(text):
    """RunResult from the text of run.txt. Raises ValueError if it is incomplete."""
    values = dict(line.split() for line in text.splitlines() if line.strip())
    return RunResult(**{field: (int(values[field]) if field in _INT_FIELDS else float(values[field]))
                        for field in RunResult._fields})


def run_experiment(params, data_folder, out_folder):
    """Train one SOM with `params` and write its files to `out_folder`.

    Data files are looked up in `data_folder`. Writes parameters.txt,
    results.txt, clustering.txt and run.txt; returns the RunResult.
    """
    start = time.time()
    data = load_train_test(params, data_folder)
    som = SOM.from_parameters(params, data.x_train)
    initial = som.quantization_error(data.x_train)
    history = train(som, data.x_train, data.x_test, params)
    labelling, acc = evaluate(som, data.x_train, data.labels_train, data.x_test, data.labels_test)
    result = RunResult(
        initial_train_error=initial,
        final_train_error=history[-1].train_error,
        final_test_error=history[-1].test_error,
        accuracy=acc.rate, correct=acc.correct, total=acc.total,
        empty_neurons=int(sum(1 for label in labelling.labels if label == NO_LABEL)),
        seconds=time.time() - start)

    os.makedirs(out_folder, exist_ok=True)
    write_results(os.path.join(out_folder, RESULTS_FILE), history)
    write_clustering(os.path.join(out_folder, CLUSTERING_FILE), som, labelling)
    with open(os.path.join(out_folder, PARAMETERS_FILE), "w", encoding="utf-8") as f:
        f.write(format_parameters(params))
    # run.txt last: its presence means the folder is complete.
    with open(os.path.join(out_folder, RUN_FILE), "w", encoding="utf-8") as f:
        f.write(format_run(result))
    return result


def load_previous(params, out_folder):
    """The RunResult saved in `out_folder` if it was made with `params`, else None."""
    try:
        with open(os.path.join(out_folder, PARAMETERS_FILE), encoding="utf-8") as f:
            if f.read() != format_parameters(params):
                return None
        with open(os.path.join(out_folder, RUN_FILE), encoding="utf-8") as f:
            return parse_run(f.read())
    except (OSError, ValueError, KeyError, TypeError):
        return None


def settings_key(params):
    """Hashable key: two experiments with the same key give the same run."""
    return tuple(params[name] for name in SCHEMA)


def plan_runs(experiments):
    """Map every experiment to the experiment that owns its run.

    Returns {name: owner name}. The first experiment with given settings
    owns the run (and its folder); later ones with the same settings share it.
    """
    owner_of_key, owner = {}, {}
    for exp in experiments:
        owner[exp.name] = owner_of_key.setdefault(settings_key(exp.params), exp.name)
    return owner


def run_all(experiments, data_folder, out_folder, jobs=1, force=False, only=None, log=print):
    """Run the experiments (once per distinct setting); return {name: RunResult}.

    Runs whose folder already has matching results are loaded instead of
    trained, unless `force`. With jobs > 1 the runs are trained in parallel
    processes. With `only` (a group name) just that group is trained; the
    other experiments are included only if their results are already saved.
    `log` gets one line per run (None for silence).
    """
    log = log or (lambda text: None)
    owner = plan_runs(experiments)
    by_name = {exp.name: exp for exp in experiments}
    wanted = {owner[exp.name] for exp in experiments if only is None or exp.group == only}
    owners = [exp.name for exp in experiments if owner[exp.name] == exp.name]

    results, todo = {}, []
    for name in owners:
        previous = None
        if not (force and name in wanted):
            previous = load_previous(by_name[name].params, os.path.join(out_folder, name))
        if previous is not None:
            results[name] = previous
            if name in wanted:
                log(f"{name}: already done, loaded from {os.path.join(out_folder, name)}")
        elif name in wanted:
            todo.append(name)

    def report(name, result):
        results[name] = result
        log(f"{name}: train QE {result.final_train_error:.6f}  test QE "
            f"{result.final_test_error:.6f}  accuracy {100 * result.accuracy:.2f}%  "
            f"({result.seconds:.0f} s)")

    if todo:
        log(f"Training {len(todo)} run(s)" + (f", {jobs} at a time" if jobs > 1 else "") + " ...")
    if jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {name: pool.submit(run_experiment, by_name[name].params, data_folder,
                                         os.path.join(out_folder, name)) for name in todo}
            for name in todo:
                report(name, futures[name].result())
    else:
        for name in todo:
            report(name, run_experiment(by_name[name].params, data_folder,
                                        os.path.join(out_folder, name)))
    return {exp.name: results[owner[exp.name]] for exp in experiments if owner[exp.name] in results}


# ---------------------------------------------------------------- summary

def _num(value):
    """Short text for a parameter value: 10.0 -> '10', 0.01 -> '0.01'."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def settings_columns(params):
    """The setting columns of a table row: grid, epochs, eta, radius, init, seed."""
    return [f"{params['gridRows']}x{params['gridCols']}",
            str(params["maxIterations"]),
            f"{_num(params['initialLearningRate'])}->{_num(params['finalLearningRate'])}",
            f"{_num(params['initialRadius'])}->{_num(params['finalRadius'])}",
            params["initMethod"],
            str(params["randomSeed"])]


def result_columns(result):
    """The result columns of a table row."""
    return [f"{result.final_train_error:.4f}", f"{result.final_test_error:.4f}",
            f"{100 * result.accuracy:.2f}", str(result.empty_neurons),
            f"{result.seconds:.0f}"]


HEADER = ["experiment", "grid", "epochs", "eta", "radius", "init", "seed",
          "train QE", "test QE", "acc %", "empty", "time s"]


def is_seed_group(rows):
    """True when the runs differ only in randomSeed (at least two seeds)."""
    if len(rows) < 2:
        return False
    def without_seed(params):
        return tuple(params[n] for n in SCHEMA if n != "randomSeed")
    first = without_seed(rows[0].params)
    seeds = {exp.params["randomSeed"] for exp in rows}
    return all(without_seed(exp.params) == first for exp in rows) and len(seeds) == len(rows)


def seed_statistics(results):
    """{field: (mean, sample std)} of train QE, test QE and accuracy over runs."""
    stats = {}
    for field in ("final_train_error", "final_test_error", "accuracy"):
        values = np.array([getattr(r, field) for r in results], dtype=float)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        stats[field] = (float(np.mean(values)), std)
    return stats


def _table(rows):
    """Text table with left-aligned first column and right-aligned others."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for k, row in enumerate(rows):
        cells = [row[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(row[1:], widths[1:])]
        lines.append("  ".join(cells).rstrip())
        if k == 0:
            lines.append("  ".join("-" * w for w in widths))
    return lines


def format_summary(experiments, results, base_name=PARAMETERS_FILE):
    """Lines of summary.txt: one table per group, in the order of the file."""
    owner = plan_runs(experiments)
    lines = [f"SOM experiments: every run is {base_name} with the changes of its line in "
             f"{EXPERIMENTS_FILE}.",
             "train QE / test QE = quantization error after the last epoch; acc % = test set "
             "clustering accuracy;",
             f"empty = neurons without training patterns ({NO_LABEL}); time s = training time "
             "of the run.",
             "A run marked '= name' has the same settings as that run and shares its result.",
             ""]
    groups = []
    for exp in experiments:
        if exp.group not in groups:
            groups.append(exp.group)
    for group in groups:
        rows = [exp for exp in experiments if exp.group == group]
        table = [HEADER]
        for exp in rows:
            label = exp.name if owner[exp.name] == exp.name else f"{exp.name} (= {owner[exp.name]})"
            table.append([label] + settings_columns(exp.params) + result_columns(results[exp.name]))
        lines.append(f"Group: {group}")
        lines.extend(_table(table))
        if is_seed_group(rows):
            stats = seed_statistics([results[exp.name] for exp in rows])
            (tr_m, tr_s), (te_m, te_s), (ac_m, ac_s) = (
                stats["final_train_error"], stats["final_test_error"], stats["accuracy"])
            lines.append(f"Mean +- std over {len(rows)} seeds: train QE {tr_m:.4f} +- {tr_s:.4f}, "
                         f"test QE {te_m:.4f} +- {te_s:.4f}, "
                         f"accuracy {100 * ac_m:.2f} +- {100 * ac_s:.2f} %")
        lines.append("")
    return lines


CSV_SETTINGS = ["gridRows", "gridCols", "maxIterations", "initialLearningRate",
                "finalLearningRate", "initialRadius", "finalRadius", "initMethod", "randomSeed"]


def format_csv(experiments, results):
    """Lines of summary.csv: a header, then one row per experiment."""
    owner = plan_runs(experiments)
    lines = [",".join(["group", "name", "same_as"] + CSV_SETTINGS + list(RunResult._fields))]
    for exp in experiments:
        r = results[exp.name]
        same = "" if owner[exp.name] == exp.name else owner[exp.name]
        values = [exp.group, exp.name, same] + [_num(exp.params[n]) for n in CSV_SETTINGS]
        values += [str(v) if f in _INT_FIELDS else f"{v:.6f}" for f, v in r._asdict().items()]
        lines.append(",".join(values))
    return lines


def write_summary(out_folder, experiments, results):
    """Write summary.txt and summary.csv; return their paths."""
    os.makedirs(out_folder, exist_ok=True)
    paths = []
    for file_name, lines in ((SUMMARY_FILE, format_summary(experiments, results)),
                             (SUMMARY_CSV, format_csv(experiments, results))):
        path = os.path.join(out_folder, file_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        paths.append(path)
    return paths


# ---------------------------------------------------------------- program

def run(folder=".", experiments_file=EXPERIMENTS_FILE, out=OUTPUT_FOLDER, jobs=1,
        force=False, only=None, log=print):
    """Read parameters.txt and the experiments file in `folder`, run, write the summary.

    `only` trains just one group; the summary still lists every experiment
    whose results are saved. Returns (all experiments, {name: RunResult}).
    """
    base = read_parameters(os.path.join(folder, PARAMETERS_FILE))
    experiments = read_experiments(os.path.join(folder, experiments_file), base)
    if only is not None and not any(exp.group == only for exp in experiments):
        raise ExperimentError(f"{experiments_file}: no experiments in group {only!r}")
    out_folder = os.path.join(folder, out)
    results = run_all(experiments, folder, out_folder, jobs=jobs, force=force, only=only, log=log)
    write_summary(out_folder, [exp for exp in experiments if exp.name in results], results)
    return experiments, results


def main(argv=None):
    """Command-line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description="Run the SOM experiments in experiments.txt.")
    parser.add_argument("--file", default=EXPERIMENTS_FILE, help="experiments file (default: %(default)s)")
    parser.add_argument("--out", default=OUTPUT_FOLDER, help="output folder (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=1, help="runs trained in parallel (default: 1)")
    parser.add_argument("--only", help="run only the experiments of this group")
    parser.add_argument("--force", action="store_true", help="retrain runs that are already done")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    start = time.time()
    try:
        _, _ = run(".", args.file, args.out, jobs=args.jobs, force=args.force, only=args.only)
    except (ParameterError, DataError, ExperimentError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    summary = os.path.join(args.out, SUMMARY_FILE)
    print(f"\nFinished in {time.time() - start:.0f} s. Summary tables in {summary} "
          f"and {os.path.join(args.out, SUMMARY_CSV)}:\n")
    with open(summary, encoding="utf-8") as f:
        print(f.read(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
