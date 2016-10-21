import numpy as np


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))


def calculate_loss_logistic_regression(y, tx, w):
    """compute the cost by negative log likelihood."""

    prediction = sigmoid(tx @ w)

    for i in range(len(y)):
        prediction[i] = np.log(prediction[i]) if y[i] == 1 else np.log(1 - prediction[i])

    return -np.sum(prediction)


def calculate_gradient_logistic_regression(y, tx, w):
    """compute the gradient of loss."""
    sig = sigmoid(tx @ w)
    minus = [sig[i] - y[i] for i in range(len(sig))]
    dot = tx.T @ minus
    return dot


def logistic_regression_helper(y, tx, gamma, max_iters, lambda_):
    w = np.zeros((tx.shape[1], 1))
    threshold = 1e-8
    loss_prev = 0
    for iter in range(max_iters):
        """
        Do one step of gradient descent using logistic regression.
        Return the loss and the updated w.
        """
        # If lambda_ is 0 then it's the loss function of logistic regression without penalty
        # else it's the penalized version of logistic regression
        loss = calculate_loss_logistic_regression(y, tx, w) + lambda_ * np.linalg.norm(w, 2)
        gradient = calculate_gradient_logistic_regression(y, tx, w)
        w -= gradient * gamma
        if iter % 1000 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

        if (loss_prev != 0) and np.abs(loss_prev - loss) < threshold:
            break
        loss_prev = loss
    return w


def logistic_regression(y, tx, gamma, max_iters):
    """ return the final w from the logistic regression """
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_=0)


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """ return the final w from the penalized logistic regression, with lambda_ as a non 0 value"""
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_)
