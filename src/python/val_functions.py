import numpy as np

from ./src.python import costs
from ./src.python import helpers

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    num_samples = y.shape[0]
    e = y - tx.dot(w)
    gradient = -1/num_samples*tx.T.dot(e)
    return gradient


def stochastic_gradient_descent(
        y, tx, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    g = 0
    w = np.zeros(tx.shape[0])
    for n_iter in range(max_epochs):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            g += compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        g = g/int(len(y)/batch_size)
        w -= gamma*g
    return w
