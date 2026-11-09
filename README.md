# Kohonen SOM (EPL442 Assignment 3)

Self-Organizing Map written from scratch in Python + NumPy (matplotlib only for plots).
It clusters the letter data from Assignment 2.

## Layout

```
code/
  parameters.txt   all run settings (read from the current folder)
  main.py          entry point
  parameters.py    reads and validates parameters.txt
  data.py          loads training.txt / test.txt, min-max normalization to [0, 1]
  som.py           SOM grid, weight initialization, winner (BMU) search,
                   neighbourhood, weight update, decay, quantization error
  training.py      training loop (epochs in random order, errors per epoch)
  clustering.py    neuron letters (majority vote of training patterns) and
                   test-set clustering accuracy, overall and per letter
  results.py       writes results.txt (epoch, training error, testing error)
                   and clustering.txt (row, col, letter, patterns, hits)
  plots.py         report plots in plots/: error curve per epoch, letter map
                   of the grid, test accuracy per letter (matplotlib)
  experiments.py   runs every line of experiments.txt and writes summary
                   tables to experiments/summary.txt and summary.csv
  experiments.txt  the comparisons (grid size, learning rate, radius,
                   initialization) and the repeated runs with other seeds
  training.txt     Assignment 2 training set (16000 patterns)
  test.txt         Assignment 2 test set (4000 patterns)
  tests/           unit tests (standard-library unittest)
README.txt         submission README (language, libraries, commands, plots)
make_zip.py        builds the Moodle archive EPL442_ex3_<AM>.zip
```

## Run

```
cd code
python3 main.py
```

Prints one progress line per epoch, then the final training/testing error
and the clustering accuracy on the test set (overall and per letter). Writes
`results.txt` and `clustering.txt` in the same folder, and the plots
`plots/error_curve.png`, `plots/letter_map.png` and `plots/letter_accuracy.png`
(skipped with a message if matplotlib is missing). To redraw the first two
from the output files only: `python3 plots.py`. With the shipped parameters
(20x20, 100 epochs) a run takes about 2.5 minutes.

Baseline (the shipped `parameters.txt`: 20x20, 100 epochs, eta 0.5 -> 0.01,
radius 10 -> 0.5, sample init, seed 42): final training error 0.2253, testing
error 0.2346, test clustering accuracy 67.65%. Its `results.txt` and
`clustering.txt` are committed in `code/`.

## Experiments

```
cd code
python3 experiments.py --jobs 4
```

Each line of `experiments.txt` is one run: `parameters.txt` with only the
parameters named on that line changed. Every run gets its own folder
`experiments/<name>/` (its parameters.txt, results.txt, clustering.txt and
run.txt), and the summary tables (one per group, with mean and standard
deviation for the seed group) go to `experiments/summary.txt` and
`experiments/summary.csv`. Runs with the same settings are trained once, and a
finished run is not trained again on the next call (add `--force` to retrain),
so adding a line only trains the new run. `--only <group>` trains one group.
`--jobs N` trains N runs in parallel; one at a time, all 11 distinct runs take
about 20-25 minutes (4 at a time: under 7 minutes).

## Tests

```
cd code
python3 -m unittest discover -s tests -v
```

## Submission zip

```
python3 make_zip.py 123456 --name "Name Surname"
```

Run from the repository root after `main.py`, with `report.pdf` next to
`README.txt`. Writes `EPL442_ex3_123456.zip` with `code/`, `README.txt` and
`report.pdf`, and fills the name and student ID into the file headers inside
the zip only.
