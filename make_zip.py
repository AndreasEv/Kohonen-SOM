"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Builds the submission archive EPL442_ex3_<AM>.zip with the layout the
assignment asks for:

    code/          source, tests, parameters.txt, training.txt, test.txt,
                   experiments.txt, results.txt and clustering.txt of the
                   baseline run, the plots and the experiment summary tables
    README.txt
    report.pdf

Run it from the repository root, after `cd code && python3 main.py` (and
optionally `python3 experiments.py`) have produced the output files:

    python3 make_zip.py 123456 --name "Name Surname"

The author placeholders at the top of every file are filled in with --name
and the AM in the copies inside the zip only; the files in the repository are
not changed. Caches and the per-run experiment folders are left out.
"""

import argparse
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))

NAME_PLACEHOLDER = "<Name Surname>"
AM_PLACEHOLDER = "<AM>"

# Files that must be in code/ (the program, its input and the baseline output).
REQUIRED_CODE_FILES = [
    "main.py", "parameters.py", "data.py", "som.py", "training.py",
    "clustering.py", "results.py", "plots.py", "experiments.py",
    "parameters.txt", "experiments.txt", "training.txt", "test.txt",
    "results.txt", "clustering.txt",
]
# Optional extras that are included when they exist.
OPTIONAL_CODE_FILES = [
    "plots/error_curve.png", "plots/letter_map.png", "plots/letter_accuracy.png",
    "experiments/summary.txt", "experiments/summary.csv",
]
# Text files whose author placeholders are filled in.
TEXT_SUFFIXES = (".py", ".txt", ".csv")


class PackageError(Exception):
    """Raised when a file the submission needs is missing."""


def collect_files(root=ROOT):
    """Return [(path on disk, path inside the zip)] for the submission.

    Raises PackageError if a required file is missing. report.pdf is not
    checked here (see build_zip).
    """
    code = os.path.join(root, "code")
    missing = [f for f in REQUIRED_CODE_FILES if not os.path.isfile(os.path.join(code, f))]
    if not os.path.isfile(os.path.join(root, "README.txt")):
        missing.append("../README.txt")
    if missing:
        raise PackageError("missing in code/: " + ", ".join(missing) +
                           " (run `python3 main.py` in code/ first for results.txt and "
                           "clustering.txt)")

    files = [(os.path.join(code, f), "code/" + f) for f in REQUIRED_CODE_FILES]
    files += [(os.path.join(code, f), "code/" + f) for f in OPTIONAL_CODE_FILES
              if os.path.isfile(os.path.join(code, f))]
    tests = os.path.join(code, "tests")
    if os.path.isdir(tests):
        for name in sorted(os.listdir(tests)):
            # test_make_zip.py tests this script, which is not in the zip.
            if name.endswith(".py") and name != "test_make_zip.py":
                files.append((os.path.join(tests, name), "code/tests/" + name))
    files.append((os.path.join(root, "README.txt"), "README.txt"))
    return files


def fill_placeholders(text, name, am):
    """Replace the author placeholders in `text` (a value of None leaves it)."""
    if name:
        text = text.replace(NAME_PLACEHOLDER, name)
    if am:
        text = text.replace(AM_PLACEHOLDER, am)
    return text


def build_zip(am, name=None, report=None, root=ROOT, out_dir=None):
    """Write EPL442_ex3_<am>.zip and return (zip path, list of warnings)."""
    files = collect_files(root)
    warnings = []
    report = report or os.path.join(root, "report.pdf")
    if os.path.isfile(report):
        files.append((report, "report.pdf"))
    else:
        warnings.append(f"report.pdf not found ({report}): the zip has no report")

    path = os.path.join(out_dir or root, f"EPL442_ex3_{am}.zip")
    left = set()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for disk, arc in files:
            if disk.endswith(TEXT_SUFFIXES):
                with open(disk, encoding="utf-8") as f:
                    text = fill_placeholders(f.read(), name, am)
                if NAME_PLACEHOLDER in text or f"Student ID: {AM_PLACEHOLDER}" in text:
                    left.add(arc)
                zf.writestr(arc, text)
            else:
                zf.write(disk, arc)
    if left:
        warnings.append("author placeholders not filled in (use --name): " +
                        ", ".join(sorted(left)))
    return path, warnings


def main(argv=None):
    """Command line entry point; return the exit code."""
    parser = argparse.ArgumentParser(description="Build EPL442_ex3_<AM>.zip for Moodle.")
    parser.add_argument("am", help="student ID (AM), used in the zip name and the file headers")
    parser.add_argument("--name", help='"Name Surname" for the file headers')
    parser.add_argument("--report", help="path of the report PDF (default: ./report.pdf)")
    args = parser.parse_args(argv)
    try:
        path, warnings = build_zip(args.am, args.name, args.report)
    except PackageError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    print(f"Wrote {path} ({len(names)} files)")
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
