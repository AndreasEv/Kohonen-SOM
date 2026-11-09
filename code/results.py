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

clustering.txt has one line per neuron, in neuron order (row by row), with
five space-separated columns:
    row  col  letter  patterns  hits
row and col are the neuron's grid position counted from 0, letter is the
label it got after training ("–" if no training pattern maps to it),
patterns is how many training patterns it wins and hits how many of those
have its letter (hits / patterns is the neuron's purity). Example line:
    3 7 A 41 35
"""

RESULTS_FILE = "results.txt"
CLUSTERING_FILE = "clustering.txt"


def format_results(history):
    """Lines of results.txt (without newlines) for a list of EpochResult."""
    return [f"{row.epoch} {row.train_error:.6f} {row.test_error:.6f}" for row in history]


def write_results(path, history):
    """Write the training history to `path` in the results.txt format."""
    with open(path, "w", encoding="utf-8") as f:
        for line in format_results(history):
            f.write(line + "\n")


def format_clustering(som, labelling):
    """Lines of clustering.txt (without newlines): one per neuron of `som`."""
    lines = []
    for j in range(som.num_neurons):
        row, col = som.position(j)
        lines.append(f"{row} {col} {labelling.labels[j]} "
                     f"{labelling.patterns[j]} {labelling.hits[j]}")
    return lines


def write_clustering(path, som, labelling):
    """Write the neuron labels to `path` in the clustering.txt format (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        for line in format_clustering(som, labelling):
            f.write(line + "\n")
