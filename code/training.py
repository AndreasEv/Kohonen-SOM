"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

The training loop. For every epoch t = 0 .. maxIterations-1:
  1. take the learning rate eta(t) and radius sigma(t) from the schedule
     (exponential decay, see som.py),
  2. present every training pattern once, in a new random order, and update
     the weights after each pattern (SOM.train_epoch),
  3. measure the quantization error on the training set and on the test set
     with the weights at the end of the epoch.
The test set is only measured, never used to change the weights.
"""

from collections import namedtuple

from som import schedule

# One row of the training history. `epoch` counts from 1 (the value written
# to results.txt); learning_rate and radius are the values used in that epoch.
EpochResult = namedtuple("EpochResult", [
    "epoch", "learning_rate", "radius", "train_error", "test_error"])


def train(som, x_train, x_test, params, on_epoch=None):
    """Train `som` for params["maxIterations"] epochs; return the history.

    x_train, x_test: normalized patterns (rows). params: the parameters.txt
    dict (maxIterations and the learning rate / radius settings are used).
    on_epoch: optional function called with each EpochResult as soon as the
    epoch ends (main.py uses it to print progress).
    Returns a list of EpochResult, one per epoch, in order.
    """
    history = []
    for t in range(params["maxIterations"]):
        eta, sigma = schedule(params, t)
        som.train_epoch(x_train, eta, sigma)
        row = EpochResult(epoch=t + 1, learning_rate=eta, radius=sigma,
                          train_error=som.quantization_error(x_train),
                          test_error=som.quantization_error(x_test))
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
    return history
