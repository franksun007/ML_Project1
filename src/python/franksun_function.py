
'''
This file implements the following functions

least squares(y, tx)
ridge regression(y, tx, lambda_)

'''

import numpy as np


def least_squares(y, tx):
    return np.linalg.solve(tx.T @ tx, tx.T @ y)


def ridge_regression(y, tx, lambda_):
    i = np.eye(tx.shape[1])
    i[0][0] = 0  # Because we don't need to penalize the first term
    return np.linalg.solve(tx.T @ tx + lambda_ * i, tx.T @ y)

