EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

A Kohonen SOM written from scratch in Python with NumPy. It clusters the
letter data of Assignment 2 (same training/test split) without using the
labels, then gives every neuron a letter and measures the clustering
accuracy on the test set.


1. Language and version
-----------------------
Python 3 (tested with Python 3.13; any Python 3.8 or newer works).
Python is interpreted, so there is no compile step.


2. Required libraries
---------------------
  numpy        all matrix operations (required)
  matplotlib   only for the plots (optional: without it the program still
               runs and prints a message instead of drawing the plots)

Install them with:
  pip3 install numpy matplotlib
(or: pip3 install -r requirements.txt from the folder above code/)

No SOM or machine learning library is used (no MiniSom, scikit-learn, ...).
The unit tests use only the standard-library unittest module.


3. Files
--------
code/
  main.py          entry point: runs one training with parameters.txt
  parameters.py    reads and validates parameters.txt
  data.py          loads training.txt / test.txt, min-max normalization
  som.py           grid, weight initialization, winner search, Gaussian
                   neighbourhood, weight update, decay of learning rate
                   and radius, quantization error
  training.py      training loop (one epoch = all patterns, random order)
  clustering.py    neuron letters (majority vote) and test accuracy
  results.py       writes results.txt and clustering.txt
  plots.py         draws the plots for the report into code/plots/
  experiments.py   runs the parameter comparisons and the seed runs
  parameters.txt   the settings of the baseline run presented in the report
  experiments.txt  the list of experiments (changes to parameters.txt)
  training.txt     training set of Assignment 2 (16000 patterns)
  test.txt         test set of Assignment 2 (4000 patterns)
  results.txt      output of the baseline run (see section 5)
  clustering.txt   output of the baseline run (see section 5)
  plots/           the plots of the baseline run
  experiments/     summary.txt / summary.csv: tables of all experiments
  tests/           unit tests
README.txt         this file
report.pdf         the report


4. How to run
-------------
The program looks for parameters.txt in the CURRENT folder and takes every
setting from it, so run it from inside code/:

  cd code
  python3 main.py

It prints one line per epoch, then the final training and testing error
(quantization error), the number of neurons without patterns, and the
clustering accuracy on the test set (overall and per letter). With the
given parameters.txt (20x20 grid, 100 epochs) a run takes about 2.5 minutes.

parameters.txt has one "name value" pair per line:
  gridRows, gridCols       size of the 2D square grid
  numInputs                number of features per pattern (16)
  maxIterations            number of epochs
  initialLearningRate      learning rate eta in the first epoch
  finalLearningRate        learning rate eta in the last epoch
  initialRadius            neighbourhood radius sigma in the first epoch
  finalRadius              neighbourhood radius sigma in the last epoch
  randomSeed               seed for the weights and the pattern order
  initMethod               "sample" (each neuron starts as a random training
                           pattern) or "random" (uniform in [0, 1))
  trainFile, testFile      the data files
finalRadius and initMethod are our own additions to the example file of
the assignment; they are explained in the report.


5. Output files
---------------
results.txt     one line per epoch: epoch, training error, testing error
                (the quantization error: mean Euclidean distance of every
                pattern to the weights of its winning neuron)
clustering.txt  one line per neuron: row, column, letter, number of
                training patterns it won, how many of them have its letter.
                A neuron that won no pattern is marked with "–".
                Example line: 3 7 A 52 40

The results.txt and clustering.txt in code/ are from the baseline run that
the report presents, with the parameters.txt in code/:
  20x20 grid, 100 epochs, eta 0.5 -> 0.01, sigma 10 -> 0.5,
  initMethod sample, randomSeed 42
  final training error 0.2253, final testing error 0.2346,
  test set clustering accuracy 67.65% (2706/4000), 10 of 400 neurons empty.


6. How the plots are made
-------------------------
main.py draws three plots with matplotlib at the end of every run and saves
them in code/plots/:
  error_curve.png      training and testing error per epoch (results.txt)
  letter_map.png       the grid with the letter of every neuron
                       (clustering.txt), shaded by how pure the neuron is
  letter_accuracy.png  test accuracy per letter and overall

To redraw the first two from results.txt and clustering.txt alone,
without training again:
  cd code
  python3 plots.py


7. Experiments (parameter comparisons and stability)
----------------------------------------------------
  cd code
  python3 experiments.py --jobs 4

Each line of experiments.txt is one run: the baseline parameters.txt with
only the parameters named on that line changed (grid size 10x10 / 15x15 /
20x20, initial learning rate 0.5 / 0.1, radius 10->0.5 / 10->1 / 5->0.5,
initialization sample / random, and seeds 42, 1, 2, 3, 4). Every run gets
its own folder experiments/<name>/ with its parameters.txt, results.txt and
clustering.txt, and the tables for the report go to
experiments/summary.txt and experiments/summary.csv (the seed table also
has the mean and standard deviation). --jobs 4 trains 4 runs at a time
(about 7 minutes in total; one at a time about 25 minutes). A finished run
is not trained again; add --force to retrain.


8. Design choices
-----------------
Normalization: min-max per feature to [0, 1], with the minimum and maximum
of the TRAINING set only; the test set uses the same values and is clipped
to [0, 1].
Winner: the neuron with the smallest Euclidean distance to the input.
Neighbourhood: Gaussian on the grid,
  h_cj(t) = exp( -||r_j - r_c||^2 / (2 sigma(t)^2) )
Weight update after every pattern:
  w_j(t+1) = w_j(t) + eta(t) h_cj(t) (x - w_j(t))
Decay: both eta and sigma decay exponentially from their initial to their
final value over the epochs t = 0 .. T-1 (T = maxIterations):
  eta(t)   = eta_0   (eta_T   / eta_0  )^(t/(T-1))
  sigma(t) = sigma_0 (sigma_T / sigma_0)^(t/(T-1))
This is the form sigma(t) = sigma_0 exp(-t/tau) of the lecture, with
tau = (T-1) / ln(sigma_0 / sigma_T), so the last epoch uses exactly the
final value. The lecture gives a linear decay eta(t) = eta_0 (1 - t/T) for
the learning rate; we use the exponential form for eta as well, which the
assignment allows, because the map orders itself in the first few epochs
with a large eta and a wide neighbourhood, and the exponential schedule
then leaves most of the epochs for small, local corrections (fine tuning)
and never lets eta reach 0.


9. Tests
--------
  cd code
  python3 -m unittest discover -s tests -v

The tests cover the parameter reader, the data loader, every SOM formula,
training on a small synthetic data set with known clusters, the labelling
and accuracy, the plots, the experiment runner and the zip script.


10. Submission archive
----------------------
After running main.py (and experiments.py) and putting report.pdf next to
this file:

  python3 make_zip.py 123456 --name "Name Surname"

(with your own student ID instead of 123456) writes EPL442_ex3_123456.zip
with code/, README.txt and report.pdf. The name and student ID are filled
into the header of every file inside the zip; the files in the repository
are not changed.
