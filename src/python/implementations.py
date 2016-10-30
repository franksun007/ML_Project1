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

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    return (-1 / N) * np.dot(tx.T, e)


def least_squares_gd(y, tx, initial_w=None, max_iters, gamma):
    """Gradient descent algorithm."""
    if initial_w is None:
        w = np.zeros(tx.shape[1])
    else:
        w = initial_w

    loss_prev = 0
    loss = 0
    threshold = 1e-8

    for n_iter in range(max_iters):

        loss = compute_loss(y, tx, w)
        # Use the gradient function to compute the gradient
        gradient = compute_gradient(y, tx, w)
        # update w with step size
        w -= gamma * gradient

        if (loss != 0 and loss_prev != 0) and (np.abs(loss_prev - loss) < threshold):
            print("Threshold reached")
            break

        loss_prev = loss

    return (w, compute_loss(y, tx, w))


def least_squares_sgd(y, tx, initial_w=None,  max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    if initial_w is None:
        w = np.zeros(tx.shape[1])
    else:
        w = initial_w

    loss_prev = 0
    loss = 0
    threshold = 1e-8
    batch_size = len(y) / 5

    for n_iter in range(max_iters):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            # Use the gradient function to compute the gradient
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            # update w with step size
            w -= gamma * gradient

        if (loss != 0 and loss_prev != 0) and (np.abs(loss_prev - loss) < threshold):
            print("Threshold reached")
            break

        loss_prev = loss

    return (w, compute_loss(y, tx, w))


def least_squares(y, tx):
    """Least square algorithm using normal equations."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return (w, compute_loss(y, tx, w))



def ridge_regression(y, tx, lambda_):
    """Ridge Regression Algorithm using normal equations."""
    i = np.eye(tx.shape[1])
    i[0][0] = 0  # Because we don't need to penalize the first term
    w = np.linalg.solve(tx.T @ tx + lambda_ * i, tx.T @ y)
    return (w, compute_loss(y, tx, w))


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


def logistic_regression_helper(y, tx, gamma, max_iters, lambda_, initial_w=None):
    """
    Helper function that will perform the core logistic regression
    algorithm with ** Gradient Descent **.
    """
    if initial_w is None:
        w = np.zeros(tx.shape[1])
    else:
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
def logistic_regression(y, tx, intial_w=None, max_iters, gamma):
    """ return the final w from the logistic regression """
    """ Algorithm is using ** gradient descent ** """
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_=0, initial_w)


def reg_logistic_regression(y, tx, lambda_, initial_w=None, max_iters, gamma):
    """ return the final w from the penalized logistic regression, with lambda_ as a non 0 value"""
    """ Algorithm is using ** gradient descent ** """
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_, initial_w)
