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

Neighbourhood. Neurons near the winner c on the grid learn too, weighted by
a Gaussian of their grid distance to c:
    h_cj(t) = exp( - ||r_j - r_c||^2 / (2 * sigma(t)^2) )
where r_j is the (row, col) position of neuron j and sigma(t) is the radius.
h is 1 for the winner, about 0.61 at distance sigma, 0.14 at 2*sigma and
0.01 at 3*sigma, so it fades smoothly instead of cutting off at a border.

Weight update. For one input x with winner c, every neuron moves towards x:
    w_j(t+1) = w_j(t) + eta(t) * h_cj(t) * ( x - w_j(t) )

Decay. The learning rate eta and the radius sigma shrink exponentially from
their initial to their final value over the epochs t = 0, 1, ..., T-1
(T = maxIterations):
    eta(t)   = eta_0   * (eta_T   / eta_0  ) ^ ( t / (T-1) )
    sigma(t) = sigma_0 * (sigma_T / sigma_0) ^ ( t / (T-1) )
so epoch 0 uses the initial value and the last epoch exactly the final one.
Exponential decay spends most epochs on small values: the map orders itself
quickly in the first epochs with a wide neighbourhood, then has many epochs
of small, local corrections (fine tuning).

Epoch. One epoch presents every training pattern exactly once, in a new
random order (a permutation drawn from the seeded generator), with the eta
and sigma of that epoch.

Quantization error. How well the map represents a set of N patterns: the
mean Euclidean distance of every pattern to the weights of its winner,
    QE = (1/N) * sum_p || x_p - w_c(p) ||
It is computed separately on the training set and on the test set.
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


def quantization_error(weights, x):
    """Quantization error of the patterns x (patterns, inputs) on the map.

        QE = (1/N) * sum_p || x_p - w_c(p) ||

    the mean Euclidean distance of each pattern to its winning neuron.
    """
    return float(np.mean(find_winners(weights, x)[1]))


def grid_sq_distances(positions, winner):
    """Squared grid distance ||r_j - r_c||^2 from neuron `winner` to every neuron.

    positions is the (neurons, 2) array from grid_positions. Returns an array
    of length `neurons`; the winner itself gets 0.
    """
    diff = positions - positions[int(winner)]
    return np.sum(diff * diff, axis=1).astype(float)


def gaussian_neighbourhood(positions, winner, radius):
    """Gaussian neighbourhood h_cj of every neuron j around the winner c.

        h_cj = exp( - ||r_j - r_c||^2 / (2 * radius^2) )

    Uses the distance on the grid (row, col), not the distance of the
    weights. Returns an array of length `neurons` with values in (0, 1];
    the winner gets exactly 1.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    return np.exp(-grid_sq_distances(positions, winner) / (2.0 * radius * radius))


def exponential_decay(initial, final, epoch, num_epochs):
    """Value of a parameter at `epoch` when it decays exponentially.

        value(t) = initial * (final / initial) ^ ( t / (T-1) ),  t = 0 .. T-1

    Epoch 0 gives `initial`, epoch T-1 gives `final`, and every epoch in
    between multiplies by the same factor. With a single epoch (T = 1) the
    initial value is used. Used for both the learning rate and the radius.
    """
    if initial <= 0 or final <= 0:
        raise ValueError("initial and final values must be > 0")
    if num_epochs < 1 or not 0 <= epoch < num_epochs:
        raise ValueError(f"epoch must be in 0..{num_epochs - 1}, got {epoch}")
    if num_epochs == 1:
        return float(initial)
    return float(initial * (final / initial) ** (epoch / (num_epochs - 1)))


def schedule(params, epoch):
    """(learning rate, radius) to use during `epoch`, from the parameters.

    Both decay exponentially from initial... to final... over maxIterations
    epochs (see exponential_decay).
    """
    t, total = epoch, params["maxIterations"]
    eta = exponential_decay(params["initialLearningRate"], params["finalLearningRate"], t, total)
    sigma = exponential_decay(params["initialRadius"], params["finalRadius"], t, total)
    return eta, sigma


def update_weights(weights, positions, x, winner, learning_rate, radius):
    """Kohonen update for one input x with winning neuron `winner` (in place).

        w_j <- w_j + eta * h_cj * (x - w_j)     for every neuron j

    with h_cj the Gaussian neighbourhood. `weights` is changed in place and
    also returned. The winner moves the most (h = 1); neurons far away on
    the grid hardly move.
    """
    h = gaussian_neighbourhood(positions, winner, radius)            # (neurons,)
    weights += learning_rate * h[:, None] * (np.asarray(x, dtype=float) - weights)
    return weights


class SOM:
    """A Kohonen map on a rows x cols square grid.

    Attributes:
      rows, cols   grid size
      num_inputs   length of every input and weight vector
      positions    (neurons, 2) grid position (row, col) of each neuron
      weights      (neurons, num_inputs) weight vector of each neuron
      rng          the seeded random generator; also shuffles the patterns
                   every epoch
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

    def neighbourhood(self, winner, radius):
        """Gaussian neighbourhood h of every neuron around `winner`."""
        return gaussian_neighbourhood(self.positions, winner, radius)

    def train_step(self, x, learning_rate, radius):
        """One learning step for one pattern x: find the winner, update the weights.

        Returns (winner index, distance before the update).
        """
        j, d = find_winner(self.weights, x)
        update_weights(self.weights, self.positions, x, j, learning_rate, radius)
        return j, d

    def train_epoch(self, x, learning_rate, radius):
        """One epoch: every row of x once, in a random order, through train_step.

        The order is a fresh permutation from the map's seeded generator, so
        it changes every epoch but the whole run can be repeated exactly.
        """
        x = np.asarray(x, dtype=float)
        for i in self.rng.permutation(len(x)):
            self.train_step(x[i], learning_rate, radius)

    def quantization_error(self, x):
        """Mean distance of the rows of x to their winning neurons (see quantization_error)."""
        return quantization_error(self.weights, x)
