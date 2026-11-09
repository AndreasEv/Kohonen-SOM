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
  training.txt     Assignment 2 training set (16000 patterns)
  test.txt         Assignment 2 test set (4000 patterns)
  tests/           unit tests (standard-library unittest)
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
from the output files only: `python3 plots.py`. With the sample parameters
(20x20, 100 epochs) a run takes about 2.5 minutes.

## Tests

```
cd code
python3 -m unittest discover -s tests -v
```
