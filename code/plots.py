"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Plots for the report, drawn from the output files of a run so they work for
any run (any grid size, any number of epochs):

  error_curve.png      training and testing quantization error per epoch
                       (from results.txt)
  letter_map.png       the grid with the letter of every neuron; the cell
                       shade is the neuron's purity = hits / patterns, i.e.
                       the share of its training patterns that have its
                       letter (from clustering.txt); "–" cells won no pattern
  letter_accuracy.png  test-set clustering accuracy for every letter, with
                       the overall accuracy as a dashed line (from the
                       Accuracy that clustering.evaluate / main.run returns,
                       because the files do not hold test results)

main.py calls save_plots() after every run and writes the images to
plots/ next to results.txt. To redraw the first two from the files alone:
    python3 plots.py [folder]

matplotlib is the only extra library and is used only here, with the
non-interactive "Agg" backend, so plotting works without a screen. It is
imported only when a plot is drawn: the rest of the program and the file
readers below run without it.
"""

import os
import sys

import numpy as np

from clustering import NO_LABEL
from results import CLUSTERING_FILE, RESULTS_FILE

PLOTS_FOLDER = "plots"
ERROR_PLOT = "error_curve.png"
MAP_PLOT = "letter_map.png"
ACCURACY_PLOT = "letter_accuracy.png"
DPI = 200

# Colours: blue and orange are easy to tell apart, also for colour-blind readers.
TRAIN_COLOUR = "#2a78d6"
TEST_COLOUR = "#eb6834"
BAR_COLOUR = "#2a78d6"
PURITY_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # light -> dark blue
EMPTY_COLOUR = "#e4e3df"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID_COLOUR = "#dddcd8"


class PlotError(Exception):
    """An output file is missing or not in the expected format."""


# ---------------------------------------------------------------- reading files

def read_results(path):
    """Read results.txt. Returns (epochs, train_errors, test_errors) as arrays."""
    if not os.path.isfile(path):
        raise PlotError(f"{path} not found")
    with open(path, encoding="utf-8") as f:
        if not f.read().strip():
            raise PlotError(f"{path} is empty")
    try:
        table = np.loadtxt(path, ndmin=2)
    except ValueError as error:
        raise PlotError(f"{path}: {error}") from None
    if table.shape[1] != 3:
        raise PlotError(f"{path}: expected 3 columns (epoch train test), found {table.shape[1]}")
    return table[:, 0].astype(int), table[:, 1], table[:, 2]


def read_clustering(path):
    """Read clustering.txt into a dict of arrays, one entry per neuron.

    Keys: rows, cols, letters, patterns, hits, shape (grid rows, grid cols).
    """
    if not os.path.isfile(path):
        raise PlotError(f"{path} not found")
    rows, cols, letters, patterns, hits = [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise PlotError(f"{path} line {number}: expected 5 columns "
                                f"(row col letter patterns hits), found {len(parts)}")
            try:
                r, c, p, h = int(parts[0]), int(parts[1]), int(parts[3]), int(parts[4])
            except ValueError:
                raise PlotError(f"{path} line {number}: row, col, patterns and hits "
                                f"must be integers") from None
            rows.append(r)
            cols.append(c)
            letters.append(parts[2])
            patterns.append(p)
            hits.append(h)
    if not rows:
        raise PlotError(f"{path} is empty")
    rows, cols = np.array(rows), np.array(cols)
    if rows.min() < 0 or cols.min() < 0:
        raise PlotError(f"{path}: negative row or column")
    return {"rows": rows, "cols": cols, "letters": np.array(letters, dtype=object),
            "patterns": np.array(patterns), "hits": np.array(hits),
            "shape": (int(rows.max()) + 1, int(cols.max()) + 1)}


# ---------------------------------------------------------------- drawing

def _pyplot():
    """Import pyplot with the file-only Agg backend (works without a screen)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    """Light grid behind the data, no top/right frame."""
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOUR, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK)


def _save(fig, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _pyplot().close(fig)
    return path


def plot_error_curve(epochs, train_errors, test_errors, path):
    """Line plot of training and testing quantization error per epoch."""
    epochs = np.asarray(epochs)
    if not (len(epochs) == len(train_errors) == len(test_errors)) or len(epochs) == 0:
        raise ValueError("epochs, train_errors and test_errors must have the same, non-zero length")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(epochs, train_errors, color=TRAIN_COLOUR, linewidth=1.8, label="Training set")
    ax.plot(epochs, test_errors, color=TEST_COLOUR, linewidth=1.8, linestyle="--", label="Test set")
    if len(epochs) == 1:     # a single point is not visible as a line
        ax.plot(epochs, train_errors, "o", color=TRAIN_COLOUR)
        ax.plot(epochs, test_errors, "o", color=TEST_COLOUR)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Quantization error")
    ax.set_title("Quantization error per epoch", color=INK)
    ax.set_xlim(epochs[0], epochs[-1] if len(epochs) > 1 else epochs[0] + 1)
    _style(ax)
    ax.legend(frameon=False, title=f"final: train {train_errors[-1]:.4f}, test {test_errors[-1]:.4f}",
              title_fontsize="small")
    return _save(fig, path)


def plot_letter_map(clustering, path):
    """The SOM grid with each neuron's letter; shade = purity (hits / patterns)."""
    from matplotlib.colors import LinearSegmentedColormap

    plt = _pyplot()
    n_rows, n_cols = clustering["shape"]
    purity = np.full((n_rows, n_cols), np.nan)
    letters = np.full((n_rows, n_cols), NO_LABEL, dtype=object)
    patterns, hits = clustering["patterns"], clustering["hits"]
    for r, c, letter, p, h in zip(clustering["rows"], clustering["cols"],
                                  clustering["letters"], patterns, hits):
        letters[r, c] = letter
        if p > 0 and letter != NO_LABEL:
            purity[r, c] = h / p

    # grey for neurons without patterns ("bad" = masked cells)
    cmap = LinearSegmentedColormap.from_list("purity", PURITY_RAMP).with_extremes(bad=EMPTY_COLOUR)
    size = max(4.0, min(9.0, 0.38 * max(n_rows, n_cols)))
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    image = ax.imshow(np.ma.masked_invalid(purity), cmap=cmap, vmin=0, vmax=1,
                      origin="upper", interpolation="nearest")
    fontsize = max(4, min(14, 170 / max(n_rows, n_cols)))
    for r in range(n_rows):
        for c in range(n_cols):
            value = purity[r, c]
            colour = "white" if not np.isnan(value) and value >= 0.55 else INK
            ax.text(c, r, letters[r, c], ha="center", va="center", fontsize=fontsize,
                    color=colour if letters[r, c] != NO_LABEL else MUTED_INK,
                    fontweight="bold" if letters[r, c] != NO_LABEL else "normal")
    # thin white lines between the cells
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    step = 1 if max(n_rows, n_cols) <= 20 else 5
    ax.set_xticks(np.arange(0, n_cols, step))
    ax.set_yticks(np.arange(0, n_rows, step))
    ax.tick_params(labelsize=7, colors=MUTED_INK, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    empty = int(np.sum(letters == NO_LABEL))
    ax.set_title(f"Letter of each neuron ({n_rows}x{n_cols} map, {empty} without patterns: {NO_LABEL})",
                 color=INK, fontsize=10)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Purity: share of the neuron's training patterns with its letter", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)
    return _save(fig, path)


def plot_letter_accuracy(per_letter, path, overall=None):
    """Bar chart of test accuracy per letter (in %), best letter first.

    per_letter maps letter -> (correct, total), as in clustering.Accuracy.
    overall (0..1) is drawn as a dashed line when given.
    """
    if not per_letter:
        raise ValueError("per_letter is empty")
    rates = {letter: (100.0 * c / n if n else 0.0) for letter, (c, n) in per_letter.items()}
    order = sorted(rates, key=lambda letter: (-rates[letter], letter))
    values = [rates[letter] for letter in order]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(4.0, 0.27 * len(order) + 1.5), 3.6))
    ax.bar(range(len(order)), values, width=0.72, color=BAR_COLOUR, label="Accuracy of the letter")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Letter")
    _style(ax)
    ax.grid(False, axis="x")
    if overall is not None:
        ax.axhline(100 * overall, color=INK, linewidth=1, linestyle="--",
                   label=f"Overall accuracy ({100 * overall:.1f}%)")
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    ax.set_title("Test-set clustering accuracy per letter", color=INK)
    return _save(fig, path)


# ---------------------------------------------------------------- all plots of a run

def save_plots(folder=".", accuracy=None):
    """Draw the plots of the run whose results.txt / clustering.txt are in `folder`.

    Images go to folder/plots/. The accuracy chart needs `accuracy` (the
    clustering.Accuracy of the run) and is skipped without it. Returns the
    list of written paths.
    """
    out = os.path.join(folder, PLOTS_FOLDER)
    written = [plot_error_curve(*read_results(os.path.join(folder, RESULTS_FILE)),
                                os.path.join(out, ERROR_PLOT)),
               plot_letter_map(read_clustering(os.path.join(folder, CLUSTERING_FILE)),
                               os.path.join(out, MAP_PLOT))]
    if accuracy is not None:
        written.append(plot_letter_accuracy(accuracy.per_letter, os.path.join(out, ACCURACY_PLOT),
                                            overall=accuracy.rate))
    return written


def main(argv=None):
    """python3 plots.py [folder]: redraw the error curve and the letter map."""
    argv = sys.argv[1:] if argv is None else argv
    folder = argv[0] if argv else "."
    try:
        written = save_plots(folder)
    except PlotError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ImportError:
        print("Error: matplotlib is not installed (pip install matplotlib)", file=sys.stderr)
        return 1
    for path in written:
        print(f"Saved {path}")
    print(f"({ACCURACY_PLOT} needs the test results; it is drawn by main.py.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
