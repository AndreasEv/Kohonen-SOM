"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Reads and validates parameters.txt. Every run setting comes from this file,
so no value is hard-coded in the rest of the program.

File format: one "name value" pair per line, separated by whitespace.
Blank lines and lines starting with '#' are ignored, and so is anything
after a '#' on a line. Every name in SCHEMA must appear exactly once.
To add a new parameter, add one entry to SCHEMA.
"""

import os


class ParameterError(ValueError):
    """Raised when parameters.txt is missing a setting or has a bad value."""


def _parse_int(text):
    """Parse a whole number. '20' is accepted, '20.5' and 'abc' are not."""
    return int(text)


def _parse_float(text):
    """Parse a real number, e.g. '0.5' or '1e-2'. Rejects nan and inf."""
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("not a finite number")
    return value


def _parse_str(text):
    """Keep the value as text (used for file names)."""
    return text


# name -> (parser, check on the parsed value, description of a valid value)
SCHEMA = {
    "gridRows":            (_parse_int,   lambda v: v >= 1, "an integer >= 1"),
    "gridCols":            (_parse_int,   lambda v: v >= 1, "an integer >= 1"),
    "numInputs":           (_parse_int,   lambda v: v >= 1, "an integer >= 1"),
    "maxIterations":       (_parse_int,   lambda v: v >= 1, "an integer >= 1"),
    "initialLearningRate": (_parse_float, lambda v: 0 < v <= 1, "a number in (0, 1]"),
    "finalLearningRate":   (_parse_float, lambda v: 0 < v <= 1, "a number in (0, 1]"),
    "initialRadius":       (_parse_float, lambda v: v > 0, "a number > 0"),
    "randomSeed":          (_parse_int,   lambda v: v >= 0, "an integer >= 0"),
    "initMethod":          (_parse_str,   lambda v: v in ("random", "sample"),
                            "'random' or 'sample'"),
    "trainFile":           (_parse_str,   lambda v: len(v) > 0, "a file name"),
    "testFile":            (_parse_str,   lambda v: len(v) > 0, "a file name"),
}


def parse_parameters(lines, source="parameters.txt"):
    """Parse the lines of a parameters file into a dict of typed values.

    `source` is only used in error messages. Raises ParameterError on an
    unknown, duplicated, missing or invalid setting, naming the line.
    """
    params = {}
    for number, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{source}, line {number}"
        parts = line.split()
        if len(parts) != 2:
            raise ParameterError(f"{where}: expected 'name value', got {line!r}")
        name, text = parts
        if name not in SCHEMA:
            known = ", ".join(SCHEMA)
            raise ParameterError(f"{where}: unknown parameter {name!r} (known: {known})")
        if name in params:
            raise ParameterError(f"{where}: {name} is given more than once")
        parser, is_valid, expected = SCHEMA[name]
        try:
            value = parser(text)
        except ValueError:
            raise ParameterError(f"{where}: {name} must be {expected}, got {text!r}") from None
        if not is_valid(value):
            raise ParameterError(f"{where}: {name} must be {expected}, got {text!r}")
        params[name] = value

    missing = [name for name in SCHEMA if name not in params]
    if missing:
        raise ParameterError(f"{source}: missing parameter(s): {', '.join(missing)}")

    # Checks that involve more than one setting.
    if params["finalLearningRate"] > params["initialLearningRate"]:
        raise ParameterError(
            f"{source}: finalLearningRate ({params['finalLearningRate']}) must not be "
            f"larger than initialLearningRate ({params['initialLearningRate']})")
    return params


def read_parameters(path):
    """Read and validate the parameters file at `path`; return a dict.

    Example: read_parameters("parameters.txt")["gridRows"] -> 20
    """
    if not os.path.isfile(path):
        raise ParameterError(f"{path}: file not found")
    with open(path, encoding="utf-8") as f:
        return parse_parameters(f, source=os.path.basename(path))
