import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))


def calculate_loss_logistic_regression(y, tx, w):
    """compute the cost by negative log likelihood."""
    prediction = sigmoid(tx @ w)
    prediction[np.where(y == 1)] = y[np.where(y == 1)] * np.log(prediction[np.where(y == 1)])
    prediction[np.where(y == 0)] = y[np.where(y == 0)] * np.log(1 - prediction[np.where(y == 0)])
    return -np.sum(prediction)


def calculate_gradient_logistic_regression(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx @ w) - y)


def calculate_hessian_logistic_regression(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate hessian: TODO
    # ***************************************************
    Snn = (sigmoid(tx @ w) * (1 - sigmoid(tx @ w)))
    Snn = Snn.reshape(Snn.shape[0])
    S = np.diag(Snn)
    return tx.T @ S @ tx


def logistic_regression(y, tx, gamma, max_iters):
    w = np.zeros((tx.shape[1], 1))
    threshold = 1e-8
    losses = []

    for iter in range(max_iters):
        """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
        loss = calculate_loss_logistic_regression(y, tx, w)
        gradient = calculate_gradient_logistic_regression(y, tx, w)
        hessian = calculate_hessian_logistic_regression(y, tx, w)
        w -= np.linalg.solve(hessian, gradient) * gamma
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w


def calculate_loss_penalized_logistic_regression(y, tx, w, lambda_):
    return calculate_loss_logistic_regression(y, tx, w) + lambda_ * np.linalg.norm(w, 2)


def reg_logistic_regression(y, tx, lambda_,gamma, max_iters):
    w = np.zeros((tx.shape[1], 1))
    threshold = 1e-8
    losses = []
    for iter in range(max_iters):
        loss = calculate_loss_penalized_logistic_regression(y, tx, w, lambda_)
        gradient = calculate_gradient_logistic_regression(y, tx, w)
        hessian = calculate_hessian_logistic_regression(y, tx, w)
        w -= np.linalg.solve(hessian, gradient) * gamma
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w
