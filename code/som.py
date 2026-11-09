"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

The SOM itself: a 2D square grid of neurons, the initialization of their
weight vectors, and the search for the winning neuron (Best Matching Unit).

Grid layout. The map has gridRows x gridCols neurons. Neuron k sits at grid
position (row, col) with k = row * gridCols + col, so neuron 0 is the top
left corner and the last neuron is the bottom right corner. The topology is
square: the neighbours of (r, c) are (r-1, c), (r+1, c), (r, c-1), (r, c+1),
and the distance between two neurons on the grid is the Euclidean distance
between their (row, col) positions.

Weights. Every neuron has a weight vector with numInputs values, stored as
one row of the array `weights` (shape: neurons x numInputs). Two ways to
initialize them are offered (parameter initMethod):
  - "random": every weight drawn uniformly from [0, 1), the same range as the
    normalized inputs.
  - "sample": every neuron starts as a copy of a training pattern picked at
    random, so the map begins inside the region the data actually occupies.
Both use the random generator seeded with randomSeed, so a run can be
repeated exactly.

Winner (BMU). For an input x the winner is the neuron j with the smallest
Euclidean distance
    d_j = || x - w_j || = sqrt( sum_i (x_i - w_ji)^2 ).
If two neurons are equally close, the one with the lower index wins.

The neighbourhood function, the weight update and the decay of the learning
rate and radius are added in a later step.
"""

import numpy as np

INIT_METHODS = ("random", "sample")

# How many patterns find_winners compares with the grid at once. It keeps the
# temporary (patterns x neurons x inputs) array small (about 13 MB for a
# 20x20 grid and 16 inputs) while still using fast NumPy operations.
_CHUNK = 256


def grid_positions(rows, cols):
    """Return the (row, col) grid position of every neuron.

    Shape (rows * cols, 2), integer. Row k is the position of neuron k, with
    k = row * cols + col.
    Example: grid_positions(2, 3) -> [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]
    """
    r, c = np.divmod(np.arange(rows * cols), cols)
    return np.stack([r, c], axis=1)


def init_random_weights(num_neurons, num_inputs, rng):
    """Random initialization: every weight uniform in [0, 1).

    The inputs are normalized to [0, 1], so this places the neurons in the
    same unit cube as the data. Returns an array (num_neurons, num_inputs).
    """
    return rng.random((num_neurons, num_inputs))


def init_sample_weights(num_neurons, x_train, rng):
    """Data-sample initialization: each neuron copies a random training pattern.

    Patterns are drawn without repetition, so no two neurons start equal
    (unless the data has duplicates). If the grid has more neurons than
    there are patterns, patterns are drawn with repetition instead.
    Returns a new array (num_neurons, num_inputs); x_train is not changed.
    """
    x_train = np.asarray(x_train, dtype=float)
    if len(x_train) == 0:
        raise ValueError("data-sample initialization needs at least one training pattern")
    replace = num_neurons > len(x_train)
    chosen = rng.choice(len(x_train), size=num_neurons, replace=replace)
    return x_train[chosen].copy()


def init_weights(method, num_neurons, num_inputs, rng, x_train=None):
    """Initialize the weights with `method` ("random" or "sample").

    x_train is only needed for "sample". Returns (num_neurons, num_inputs).
    """
    if method == "random":
        return init_random_weights(num_neurons, num_inputs, rng)
    if method == "sample":
        if x_train is None:
            raise ValueError("initMethod 'sample' needs the training data")
        x_train = np.asarray(x_train, dtype=float)
        if x_train.ndim != 2 or x_train.shape[1] != num_inputs:
            raise ValueError(f"training data must have {num_inputs} columns, "
                             f"got shape {x_train.shape}")
        return init_sample_weights(num_neurons, x_train, rng)
    raise ValueError(f"unknown initMethod {method!r} (use one of {', '.join(INIT_METHODS)})")


def find_winner(weights, x):
    """Return (index, distance) of the winning neuron for one input vector x.

    Winner: j = argmin_j || x - w_j ||  (Euclidean distance).
    Ties go to the neuron with the lowest index.
    """
    diff = weights - np.asarray(x, dtype=float)       # (neurons, inputs)
    dist = np.sqrt(np.sum(diff * diff, axis=1))       # || x - w_j || for every j
    j = int(np.argmin(dist))
    return j, float(dist[j])


def find_winners(weights, x):
    """Winners for many input vectors at once.

    x has shape (patterns, inputs). Returns (indices, distances), both of
    length `patterns`: the winning neuron of each pattern and its Euclidean
    distance to that neuron. Gives exactly the same answer as calling
    find_winner on every row, only faster. The mean of `distances` is the
    quantization error.
    """
    x = np.asarray(x, dtype=float)
    indices = np.empty(len(x), dtype=int)
    distances = np.empty(len(x))
    for start in range(0, len(x), _CHUNK):
        block = x[start:start + _CHUNK]
        diff = block[:, None, :] - weights[None, :, :]   # (chunk, neurons, inputs)
        dist = np.sqrt(np.sum(diff * diff, axis=2))      # (chunk, neurons)
        best = np.argmin(dist, axis=1)
        indices[start:start + len(block)] = best
        distances[start:start + len(block)] = dist[np.arange(len(block)), best]
    return indices, distances


class SOM:
    """A Kohonen map on a rows x cols square grid.

    Attributes:
      rows, cols   grid size
      num_inputs   length of every input and weight vector
      positions    (neurons, 2) grid position (row, col) of each neuron
      weights      (neurons, num_inputs) weight vector of each neuron
      rng          the seeded random generator, kept for later random steps
                   (e.g. shuffling the patterns every epoch)
    """

    def __init__(self, rows, cols, num_inputs, seed, init_method="random", x_train=None):
        """Build the grid and initialize the weights with `init_method`."""
        if rows < 1 or cols < 1 or num_inputs < 1:
            raise ValueError("rows, cols and num_inputs must be at least 1")
        self.rows = rows
        self.cols = cols
        self.num_inputs = num_inputs
        self.rng = np.random.default_rng(seed)
        self.positions = grid_positions(rows, cols)
        self.weights = init_weights(init_method, rows * cols, num_inputs, self.rng, x_train)

    @classmethod
    def from_parameters(cls, params, x_train):
        """Build a SOM from the parameters.txt settings and the normalized training data."""
        return cls(params["gridRows"], params["gridCols"], params["numInputs"],
                   seed=params["randomSeed"], init_method=params["initMethod"],
                   x_train=x_train)

    @property
    def num_neurons(self):
        """Number of neurons on the grid (rows * cols)."""
        return self.rows * self.cols

    def position(self, index):
        """Grid (row, col) of neuron `index`."""
        return divmod(int(index), self.cols)

    def index(self, row, col):
        """Neuron index of grid position (row, col)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"({row}, {col}) is outside the {self.rows}x{self.cols} grid")
        return row * self.cols + col

    def winner(self, x):
        """(index, distance) of the winning neuron for one input x."""
        return find_winner(self.weights, x)

    def winners(self, x):
        """(indices, distances) of the winning neurons for every row of x."""
        return find_winners(self.weights, x)
