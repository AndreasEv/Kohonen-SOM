"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Entry point. Looks for parameters.txt in the current working folder and
runs one SOM training with the settings found there:
  read parameters -> load and normalize the data -> build and initialize the
  map -> train for maxIterations epochs -> write results.txt -> print the
  final training / testing error.
"""

import os
import sys
import time

from data import DataError, load_train_test
from parameters import ParameterError, read_parameters
from results import RESULTS_FILE, write_results
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


def run(folder=".", verbose=True):
    """Train with folder/parameters.txt and write folder/results.txt.

    Data files are looked up in `folder` too. Returns (som, data, history).
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
    return som, data, history


def main():
    """Run the program from the current folder; return the exit code."""
    path = find_parameters_file()
    if path is None:
        print(f"Error: {PARAMETERS_FILE} not found in {os.getcwd()}", file=sys.stderr)
        return 1
    start = time.time()
    try:
        _, _, history = run(".")
    except (ParameterError, DataError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    last = history[-1]
    print(f"\nFinished in {time.time() - start:.1f} s. Per-epoch errors saved to {RESULTS_FILE}.")
    print(f"Final training error (QE): {last.train_error:.6f}")
    print(f"Final testing error (QE):  {last.test_error:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
