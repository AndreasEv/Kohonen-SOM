"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Neuron labels and clustering accuracy. This is the only place where the
letters are used, and only after training has finished.

Labelling. Every training pattern is mapped to its winning neuron. A neuron
gets the letter that appears most often among the training patterns it wins:
    label(j) = argmax_L  #{ p : winner(p) = j and letter(p) = L }
If two letters are equally frequent, the one that comes first in the
alphabet is chosen, so the result never depends on the order of the data.
A neuron that wins no training pattern gets the mark "–" (NO_LABEL).

Accuracy. A test pattern is classified with the letter of its winning
neuron. It counts as correct when that letter equals its real letter. A
test pattern whose winner has no label ("–") counts as wrong.
    accuracy = (correct test patterns) / (all test patterns)
It is also reported for every letter separately (correct / patterns of that
letter in the test set).
"""

from collections import namedtuple

import numpy as np

NO_LABEL = "–"   # en dash, the mark the assignment uses for neurons without patterns

# Result of label_neurons. All arrays have one entry per neuron.
#   labels      the letter of each neuron, or NO_LABEL
#   patterns    how many training patterns the neuron wins
#   hits        how many of them have the neuron's letter (0 for NO_LABEL)
#   letters     the sorted letters that occur in the training labels
#   counts      (neurons, letters) array: counts[j, k] = training patterns of
#               letters[k] won by neuron j
Labelling = namedtuple("Labelling", ["labels", "patterns", "hits", "letters", "counts"])

# Result of accuracy().
#   correct, total   test patterns classified correctly / all test patterns
#   rate             correct / total (0..1)
#   per_letter       {letter: (correct, total)} for every letter in the test set
Accuracy = namedtuple("Accuracy", ["correct", "total", "rate", "per_letter"])


def label_neurons(winners, labels, num_neurons):
    """Give every neuron the majority letter of the training patterns it wins.

    winners: winning neuron index of each training pattern.
    labels:  real letter of each training pattern (same length).
    Returns a Labelling. Ties go to the alphabetically first letter; neurons
    that win nothing are marked NO_LABEL.
    """
    winners = np.asarray(winners, dtype=int)
    labels = np.asarray(labels)
    if len(winners) != len(labels):
        raise ValueError(f"{len(winners)} winners but {len(labels)} labels")
    if len(winners) and (winners.min() < 0 or winners.max() >= num_neurons):
        raise ValueError(f"winner index outside 0..{num_neurons - 1}")

    letters, letter_index = np.unique(labels, return_inverse=True)   # sorted A..Z
    counts = np.zeros((num_neurons, len(letters)), dtype=int)
    np.add.at(counts, (winners, letter_index), 1)                    # vote table

    patterns = counts.sum(axis=1)
    # argmax returns the first maximum, i.e. the alphabetically first letter on a tie.
    best = np.argmax(counts, axis=1) if len(letters) else np.zeros(num_neurons, dtype=int)
    neuron_labels = np.full(num_neurons, NO_LABEL, dtype=object)
    hits = np.zeros(num_neurons, dtype=int)
    used = patterns > 0
    neuron_labels[used] = letters[best[used]]
    hits[used] = counts[used, best[used]]
    return Labelling(labels=neuron_labels, patterns=patterns, hits=hits,
                     letters=[str(l) for l in letters], counts=counts)


def classify(winners, neuron_labels):
    """Letter given to each pattern: the label of its winning neuron (maybe NO_LABEL)."""
    return np.asarray(neuron_labels, dtype=object)[np.asarray(winners, dtype=int)]


def accuracy(predicted, real):
    """Share of patterns whose predicted letter equals the real letter.

    Returns an Accuracy with the overall result and one (correct, total)
    pair per letter of `real`. A NO_LABEL prediction is always wrong.
    """
    predicted = np.asarray(predicted, dtype=object)
    real = np.asarray(real, dtype=object)
    if len(predicted) != len(real):
        raise ValueError(f"{len(predicted)} predictions but {len(real)} real letters")
    right = predicted == real
    per_letter = {}
    for letter in sorted(set(real.tolist())):
        mask = real == letter
        per_letter[str(letter)] = (int(right[mask].sum()), int(mask.sum()))
    total = len(real)
    correct = int(right.sum())
    return Accuracy(correct=correct, total=total,
                    rate=correct / total if total else 0.0, per_letter=per_letter)


def evaluate(som, x_train, labels_train, x_test, labels_test):
    """Label the trained map with the training set, then score the test set.

    Returns (Labelling, Accuracy). The weights are not changed.
    """
    train_winners, _ = som.winners(x_train)
    labelling = label_neurons(train_winners, labels_train, som.num_neurons)
    test_winners, _ = som.winners(x_test)
    return labelling, accuracy(classify(test_winners, labelling.labels), labels_test)
