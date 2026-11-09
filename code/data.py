"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Reads the letter data of Assignment 2 and normalizes it to [0, 1].

Data format: one pattern per line, "T,2,8,3,...,8": the letter first, then
numInputs comma-separated integer features. Blank lines are skipped.

The letter is returned in a separate array and is never part of the input
vectors, so the SOM cannot see it while training. It is only used after
training, to give each neuron a letter and to score the test set.

Normalization is min-max per feature:  x' = (x - min) / (max - min),
where min and max come from the TRAINING set only. The same min and max are
then applied to the test set, so no information from the test set leaks
into training. A test value outside the training range is clipped to
[0, 1], so every input lies in the same unit cube as the SOM weights.
A feature that is constant in the training set carries no information and
is mapped to 0.
"""

import os
from collections import namedtuple

import numpy as np


class DataError(ValueError):
    """Raised when a data file is missing or has a malformed line."""


# What load_train_test returns. x_* are normalized float arrays of shape
# (patterns, numInputs); labels_* are arrays of letters (str), one per row.
Dataset = namedtuple("Dataset", [
    "x_train", "labels_train", "x_test", "labels_test", "feature_min", "feature_max"])


def parse_data(lines, num_inputs, source="data"):
    """Parse data lines into (features, labels).

    features: float array of shape (patterns, num_inputs), raw values.
    labels:   array of the letters, one per pattern.
    `source` is only used in error messages. Raises DataError naming the line
    if a line has the wrong number of values, a bad label or a non-number.
    """
    rows, labels = [], []
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        where = f"{source}, line {number}"
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != num_inputs + 1:
            raise DataError(f"{where}: expected a letter and {num_inputs} values, "
                            f"got {len(parts)} fields")
        label = parts[0]
        if len(label) != 1 or not label.isalpha():
            raise DataError(f"{where}: first field must be a letter, got {label!r}")
        try:
            values = [float(p) for p in parts[1:]]
        except ValueError:
            raise DataError(f"{where}: features must be numbers, got {line!r}") from None
        if not all(np.isfinite(values)):
            raise DataError(f"{where}: features must be finite numbers")
        rows.append(values)
        labels.append(label.upper())
    if not rows:
        raise DataError(f"{source}: no data")
    return np.array(rows, dtype=float), np.array(labels)


def load_data(path, num_inputs):
    """Read the data file at `path`; return (features, labels) as in parse_data."""
    if not os.path.isfile(path):
        raise DataError(f"{path}: file not found")
    with open(path, encoding="utf-8") as f:
        return parse_data(f, num_inputs, source=os.path.basename(path))


def fit_min_max(x):
    """Return the per-feature (min, max) of `x`. Call it on the training set only."""
    return x.min(axis=0), x.max(axis=0)


def normalize(x, feature_min, feature_max):
    """Min-max normalize `x` to [0, 1] with the given per-feature min and max.

    x' = (x - min) / (max - min), clipped to [0, 1]. A feature with
    max == min is set to 0. Returns a new array; `x` is not changed.
    """
    span = feature_max - feature_min
    safe_span = np.where(span > 0, span, 1.0)  # avoid dividing by zero
    scaled = (x - feature_min) / safe_span
    scaled[:, span <= 0] = 0.0
    return np.clip(scaled, 0.0, 1.0)


def load_train_test(params, folder="."):
    """Load and normalize the training and test files named in `params`.

    `params` is the dict from parameters.read_parameters. Files are looked
    up in `folder`. Normalization statistics come from the training set only.
    """
    n = params["numInputs"]
    x_train_raw, labels_train = load_data(os.path.join(folder, params["trainFile"]), n)
    x_test_raw, labels_test = load_data(os.path.join(folder, params["testFile"]), n)
    feature_min, feature_max = fit_min_max(x_train_raw)
    return Dataset(
        x_train=normalize(x_train_raw, feature_min, feature_max),
        labels_train=labels_train,
        x_test=normalize(x_test_raw, feature_min, feature_max),
        labels_test=labels_test,
        feature_min=feature_min,
        feature_max=feature_max,
    )
