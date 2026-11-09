"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Writes the output files of a training run.

results.txt has one line per epoch with three space-separated columns:
    epoch  training_error  testing_error
where the epoch counter starts at 1 and both errors are the quantization
error (mean distance of each pattern to its winning neuron) at the end of
that epoch. Example line:  1 0.512345 0.514321
The file can be read back with numpy.loadtxt for the plots.
"""

RESULTS_FILE = "results.txt"


def format_results(history):
    """Lines of results.txt (without newlines) for a list of EpochResult."""
    return [f"{row.epoch} {row.train_error:.6f} {row.test_error:.6f}" for row in history]


def write_results(path, history):
    """Write the training history to `path` in the results.txt format."""
    with open(path, "w", encoding="utf-8") as f:
        for line in format_results(history):
            f.write(line + "\n")
