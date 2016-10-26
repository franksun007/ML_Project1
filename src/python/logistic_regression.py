import numpy as np
from helpers import *
from proj1_helpers import *


np.seterr(over='ignore')

BINARY_CLASSIFICATOIN_0 = -1
BINARY_CLASSIFICATOIN_1 = 1


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))


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

    w_max = w
    performance = 0
    i = 0

    threshold = 1e-8
    loss_prev = 0
    batch_size = 1000

    for iter in range(max_iters):

        flag = 1

        # for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):

        loss = calculate_loss_logistic_regression(y, tx, w) + lambda_ * np.linalg.norm(w, 2)
        gradient = calculate_gradient_logistic_regression(y, tx, w)
        w -= (gradient * gamma).reshape(w.shape)

        if (loss_prev != 0) and np.abs(loss_prev - loss) < threshold:
            print("Reached Theshold, exit")
            flag = 0
            break

        loss_prev = loss

        if (iter % 10) == 0:
            compare_pred = predict_labels(w, tx)
            compare_pred -= y.reshape([len(y), 1])
            nonzero = 0
            for j in range(len(compare_pred)):
                if (compare_pred[j] != 0):
                    nonzero += 1

            cur_perf = 1 - nonzero / compare_pred.size
            if cur_perf > performance:
                performance = cur_perf
                w_max = w
                i = iter

        if (iter % 300) == 0:
            print(w_max)
            print("Performance: ", performance)
            print("Iteration: ", i)


        if (iter % 100) == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

        if flag == 0:
            break

    return w


def logistic_regression(y, tx, gamma, max_iters):
    """ return the final w from the logistic regression """
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_=0)


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """ return the final w from the penalized logistic regression, with lambda_ as a non 0 value"""
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_)
