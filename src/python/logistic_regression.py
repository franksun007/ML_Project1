import numpy as np
from helpers import *


np.seterr(over='ignore')

BINARY_CLASSIFICATOIN_0 = -1
BINARY_CLASSIFICATOIN_1 = 1


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (np.exp(0) + np.exp(-t))


def calculate_loss_logistic_regression(y, tx, w):
    """compute the cost by negative log likelihood."""
    prediction = tx @ w
    y1 = np.where(y == BINARY_CLASSIFICATOIN_1)

    over_700 = np.where(prediction >= 700)

    prediction_result = np.log(1 + np.exp(prediction))
    prediction_result[over_700] = prediction[over_700]
    prediction_result[y1] -= prediction[y1]

    result = np.sum(prediction_result)
    return result


def calculate_gradient_logistic_regression(y, tx, w):
    """compute the gradient of loss."""

    y1 = np.where(y == BINARY_CLASSIFICATOIN_1)
    sig = sigmoid(tx @ w).reshape(len(y))
    sig[y1] -= y[y1]

    return tx.T @ sig


def logistic_regression_helper(y, tx, gamma, max_iters, lambda_):
    w = np.zeros((tx.shape[1], 1))
    threshold = 1e-8
    loss = 0
    loss_prev = 0
    batch_size = 5000

    # minibatch_y = y
    # minibatch_tx = tx

    for iter in range(max_iters):

        # for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):


        """
        Do one step of gradient descent using logistic regression.
        Return the loss and the updated w.
        """
        # If lambda_ is 0 then it's the loss function of logistic regression without penalty
        # else it's the penalized version of logistic regression

        loss = calculate_loss_logistic_regression(y, tx, w) + lambda_ * np.linalg.norm(w, 2)
        gradient = calculate_gradient_logistic_regression(y, tx, w)
        w -= (gradient * gamma).reshape(w.shape)

        if (iter % 10) == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

        if (loss_prev != 0) and np.abs(loss_prev - loss) < threshold:
            break

        if (iter % 5000) == 0:
            print(w)

        loss_prev = loss

    return w


def logistic_regression(y, tx, gamma, max_iters):
    """ return the final w from the logistic regression """
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_=0)


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """ return the final w from the penalized logistic regression, with lambda_ as a non 0 value"""
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_)
