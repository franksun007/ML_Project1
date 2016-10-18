import numpy as np
from src.python.costs import *
from src.python.helpers import *

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    return (-1 / N) * np.dot(tx.T, e)


def least_squares_gd(y, tx, max_iters, gamma):
    """Gradient descent algorithm."""
    w = np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradient

    return w


def least_squares_sgd(y, tx, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = np.zeros(tx.shape[1])
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 42):  # TODO: choose batch size
        gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        w -= gamma * gradient

    return w
