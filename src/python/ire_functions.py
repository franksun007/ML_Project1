import numpy as np
from src.python.costs import *


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    return (-1 / N) * np.dot(tx.T, e)


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradient

    return w
