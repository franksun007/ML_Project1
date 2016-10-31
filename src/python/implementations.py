'''
Author: Group #23
            272785
            271691
            204988

This file implements the following functions

least_squares_GD(y, tx, initial_w, max_iters, gamma)
        Linear_regression_using_gradient_descent

least_squares_SGD(y, tx, initial_w, max_iters, gamma)
        Linear_regression_using_stochastic_gradient_descent

least_squares(y, tx)
        Least_squares_regression_using_normal_equations

ridge_regression(y, tx, lambda_)
        Ridge_regression_using_normal_equations

logistic_regression(y, tx, initial_w, max_iters, gamma)
        Logistic_regression_using_gradient_descent

reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
        Regularized_logistic_regression_using_gradient_descent
'''

import numpy as np
from costs import *
from helpers import *


def compute_loss_mem_aware(y, tx, w):
    """Calculate the loss.

    Calculate the loss using mse
    The function forced reshaping of the matrix so that the boardcasting step
    will not generate memory error
    """
    dot = tx.dot(w)
    e = y.reshape((len(y), 1)) - dot.reshape((len(y), 1))
    return calculate_mse(e)


def compute_gradient_mem_aware(y, tx, w):
    """Compute the gradient.

    The function forced reshaping of the matrix so that the boardcasting step
    will not generate memory error
    """
    N = y.shape[0]
    dot = np.dot(tx, w)
    e = y.reshape((len(y), 1)) - dot.reshape((len(y), 1))
    gradient = (-1 / N) * (tx.T @ e)
    return gradient


def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w = initial_w

    loss_prev = 0
    loss = 0
    threshold = 1e-8

    for n_iter in range(max_iters):
        # Compute the gradient
        loss = compute_loss_mem_aware(y, tx, w)
        # Use the gradient function to compute the gradient
        gradient = compute_gradient_mem_aware(y, tx, w).reshape(w.shape)
        # update w with step size
        w -= gamma * gradient

        # Iterate until converge to the threshold
        if (loss != 0 and loss_prev != 0) and (np.abs(loss_prev - loss) < threshold):
            print("Threshold reached")
            break

        if (n_iter % 100) == 0:
            print("Current iteration={i}, the loss={l}".format(i=n_iter, l=loss))

        loss_prev = loss

    return (w, loss)


def least_squares_sgd(y, tx, initial_w,  max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w

    loss_prev = 0
    loss = 0
    threshold = 1e-8
    batch_size = len(y) / 5

    for n_iter in range(max_iters):
        # Using batch_iter to perform stochastic gradient descent
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute the gradient
            loss = compute_loss_mem_aware(minibatch_y, minibatch_tx, w)
            # Use the gradient function to compute the gradient
            gradient = compute_gradient_mem_aware(minibatch_y, minibatch_tx, w).reshape(w.shape)
            # update w with step size
            w -= gamma * gradient

        # Iterate until converge to the threshold
        if (loss != 0 and loss_prev != 0) and (np.abs(loss_prev - loss) < threshold):
            print("Threshold reached")
            break

        if (n_iter % 100) == 0:
            print("Current iteration={i}, the loss={l}".format(i=n_iter, l=loss))

        loss_prev = loss

    return (w, loss)


def least_squares(y, tx):
    """Least square algorithm using normal equations."""
    # trying to find the result for (X.T @ X)^(-1) @ (X.T @ y)
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return (w, compute_loss_mem_aware(y, tx, w))


def ridge_regression(y, tx, lambda_):
    """Ridge Regression Algorithm using normal equations."""
    i = np.eye(tx.shape[1])
    i[0][0] = 0  # Because we don't need to penalize the first term
    # Trying to find the result for (X.T @ X + lambda_ I)^(-1) @ (X.T @ y)
                                    # penalized term
    w = np.linalg.solve(tx.T @ tx + lambda_ * i, tx.T @ y)
    return (w, compute_loss_mem_aware(y, tx, w))


####################################################################

# Constant to indicate +1 and 0 for classification
BINARY_CLASSIFICATION_1 = 1

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))


def calculate_loss_logistic_regression(y, tx, w):
    """compute the cost by negative log likelihood."""
    prediction = tx @ w

    y1 = np.where(y == BINARY_CLASSIFICATION_1)
    prediction_result = np.log(1 + np.exp(prediction))
    # only -y when classification result is 1
    prediction_result[y1] -= prediction[y1]

    result = np.sum(prediction_result)
    return result


def calculate_gradient_logistic_regression(y, tx, w):
    """compute the gradient of loss."""
    y1 = np.where(y == BINARY_CLASSIFICATION_1)
    sig = sigmoid(tx @ w).reshape(len(y))
    # only -y when classification result is 1
    sig[y1] -= y[y1]

    return (tx.T @ sig).reshape((tx.shape[1], 1))


def logistic_regression_helper(y, tx, initial_w, max_iters, gamma, lambda_):
    """
    Helper function that will perform the core logistic regression
    algorithm with ** Gradient Descent **.
    """

    w = initial_w

    threshold = 1e-8    # Threshold for converge
    loss_prev = 0       # the previous loss
    loss = 0

    for iter in range(max_iters):
        # lambda_ = 0 if performing pure logistic regression
        loss = calculate_loss_logistic_regression(y, tx, w) + lambda_ * np.linalg.norm(w, 2)
        gradient = calculate_gradient_logistic_regression(y, tx, w)

        w -= gradient * gamma

        # If converge
        if (loss_prev != 0 and loss != 0) and np.abs(loss_prev - loss) < threshold:
            print("Reached Theshold, exit")
            break

        loss_prev = loss
        if (iter % 100) == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

    return (w, loss)


"""
    Logistic Regression and Regularized Logisitic Regression
    share the same core code. The only difference is that
    for logistic regression, lambda_, the regularization term
    is set to 0.
"""
def logistic_regression(y, tx, intial_w, max_iters, gamma):
    """ return the final w from the logistic regression """
    """ Algorithm is using ** gradient descent ** """
    return logistic_regression_helper(y, tx, intial_w, max_iters, gamma, 0)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ return the final w from the penalized logistic regression, with lambda_ as a non 0 value"""
    """ Algorithm is using ** gradient descent ** """
    return logistic_regression_helper(y, tx, initial_w, max_iters, gamma, lambda_)
