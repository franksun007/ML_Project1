
'''
This file implements the following functions

least squares GD (y, tx, gamma, max iters)
least squares SGD(y, tx, gamma, max iters)
least squares(y, tx)
ridge regression(y, tx, lambda_)
logistic regression(y, tx, gamma, max iters) # Logistic regression using gradient descent or SGD
reg logistic regression(y, tx, lambda_, gamma, max iters)

'''

import numpy as np
from src.python.helpers import batch_iter


def compute_mse(y, tx, w):
    """
    Calculate the mse.
    """
    # compute loss by MSE / MAE
    N = len(y)
    e = y - np.dot(tx, w)
    return  1.0 / 2 / N * np.dot(e.T, e)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.dot(tx, w)
    result = - 1 / len(y) * np.dot(tx.T, e)
    return result


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
        # store w and loss
    return w


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    # implement stochastic gradient computation.
    e = y - np.dot(tx, w)
    return - 1 / len(y) * np.dot(tx.T, e)


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""

    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_epochs):
        # compute gradient and mse
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            # update w by gradient
            w = w - gamma * gradient
    return w


# Reference to gradient_descent function
def least_squares_gd(y, tx, gamma, max_iters):
    return 0


# Reference to stochastic_gradient_descent function
def least_squares_sgd(y, tx, gamma, max_iters):
    return 0


def least_squares(y, tx):
    return np.linalg.solve(tx.T @ tx, tx.T @ y)


def ridge_regression(y, tx, lambda_):
    i = np.eye(tx.shape[1])
    i[0][0] = 0  # Because we don't need to penalize the first term
    return np.linalg.solve(tx.T @ tx + lambda_ * i, tx.T @ y)


# No idea why this function is here
# Aiming to return the negative of the value of log-likelihood
def compute_cost_logistic_regression(w_guess, y, tx):
    nll = np.log(1 + np.exp(-y @ w_guess.T @ tx))
    return nll


# I would assume that this function actually implemented
# the computation part of the gradient of logistic regression.
#
# TODO: UNTESTED
# I have no idea whether it's correct or not, but we will figure it
# out this week during the exercise section.
#
# This function supposed to call the above function.
# However, I didn't invoke it. (It's according to last year's lab sheet etc.)
#
# This function implements the below equation:
# g = \delta L / \delta w = -X.T [ sigma(Xw) - t ]
#
# Return the negative value of the gradient of the log-likelihood
def compute_gradient_neg_log_likelihood(w_guess, y, tx):
    g = -tx.T @ ((np.exp(tx) / (1 + np.exp(tx @ w_guess))) - y)
    return g


def logistic_regression(y, tx, gamma, max_iters):
    w = 0  # TODO: I have no idea what the init value should be
    # This function is following the fashion of gradient descent,
    # we might factor out the code, but let's figure out if this
    # function will work first.
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient_neg_log_likelihood(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
        # store w and loss
    return w


# I barely have an idea on how to implement the regularized version of
# logistic regression..
def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    return 0
