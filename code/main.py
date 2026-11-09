"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Entry point. Looks for parameters.txt in the current working folder and
runs one SOM training with the settings found there:
  read parameters -> load and normalize the data -> build and initialize the
  map -> train for maxIterations epochs -> write results.txt -> label every
  neuron with the majority letter of its training patterns and write
  clustering.txt -> print the final training / testing error and the
  clustering accuracy on the test set (overall and per letter) -> save the
  plots for the report in plots/ (see plots.py).
"""

import os
import sys
import time

from clustering import NO_LABEL, evaluate
from data import DataError, load_train_test
from parameters import ParameterError, read_parameters
from results import CLUSTERING_FILE, RESULTS_FILE, write_clustering, write_results
from som import SOM
from training import train

PARAMETERS_FILE = "parameters.txt"


def find_parameters_file(folder="."):
    """Return the path of parameters.txt in `folder`, or None if it is missing."""
    path = os.path.join(folder, PARAMETERS_FILE)
    return path if os.path.isfile(path) else None


def print_epoch(row, total):
    """Progress line for one finished epoch."""
    print(f"Epoch {row.epoch:4d}/{total}  eta {row.learning_rate:.4f}  "
          f"radius {row.radius:.3f}  train QE {row.train_error:.6f}  "
          f"test QE {row.test_error:.6f}", flush=True)


def print_summary(history, labelling, acc):
    """Final errors, neuron labels and test accuracy (overall and per letter)."""
    last = history[-1]
    print(f"Final training error (QE): {last.train_error:.6f}")
    print(f"Final testing error (QE):  {last.test_error:.6f}")
    empty = sum(1 for label in labelling.labels if label == NO_LABEL)
    print(f"Neurons with a letter: {len(labelling.labels) - empty}/{len(labelling.labels)} "
          f"({empty} without patterns, marked {NO_LABEL})")
    print(f"Test set clustering accuracy: {100 * acc.rate:.2f}% "
          f"({acc.correct}/{acc.total})")
    print("Per letter (correct/total):")
    items = [f"{letter} {100 * c / n:5.1f}% ({c}/{n})" for letter, (c, n) in acc.per_letter.items()]
    for start in range(0, len(items), 4):
        print("  " + "   ".join(items[start:start + 4]))


def print_plots(acc, folder="."):
    """Save the report plots; a missing matplotlib only skips them."""
    try:
        import plots
        written = plots.save_plots(folder, acc)
    except ImportError:
        print("matplotlib is not installed, so no plots were drawn (pip install matplotlib).")
        return []
    print("Plots saved: " + ", ".join(written))
    return written


def run(folder=".", verbose=True):
    """Train with folder/parameters.txt; write folder/results.txt and folder/clustering.txt.

    Data files are looked up in `folder` too. Returns (som, data, history,
    labelling, accuracy); see clustering.py for the last two.
    """
    params = read_parameters(os.path.join(folder, PARAMETERS_FILE))
    data = load_train_test(params, folder)
    som = SOM.from_parameters(params, data.x_train)
    if verbose:
        print(f"{params['gridRows']}x{params['gridCols']} map, "
              f"{len(data.x_train)} training / {len(data.x_test)} test patterns, "
              f"{params['maxIterations']} epochs, initMethod {params['initMethod']}, "
              f"seed {params['randomSeed']}")
        print(f"Before training: train QE {som.quantization_error(data.x_train):.6f}  "
              f"test QE {som.quantization_error(data.x_test):.6f}", flush=True)

    total = params["maxIterations"]
    report = (lambda row: print_epoch(row, total)) if verbose else None
    history = train(som, data.x_train, data.x_test, params, on_epoch=report)
    write_results(os.path.join(folder, RESULTS_FILE), history)
    # Letters are used only now, after training.
    labelling, acc = evaluate(som, data.x_train, data.labels_train, data.x_test, data.labels_test)
    write_clustering(os.path.join(folder, CLUSTERING_FILE), som, labelling)
    return som, data, history, labelling, acc


def main():
    """Run the program from the current folder; return the exit code."""
    path = find_parameters_file()
    if path is None:
        print(f"Error: {PARAMETERS_FILE} not found in {os.getcwd()}", file=sys.stderr)
        return 1
    start = time.time()
    try:
        _, _, history, labelling, acc = run(".")
    except (ParameterError, DataError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"\nFinished in {time.time() - start:.1f} s. Per-epoch errors saved to {RESULTS_FILE}, "
          f"neuron letters to {CLUSTERING_FILE}.")
    print_summary(history, labelling, acc)
    print_plots(acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
