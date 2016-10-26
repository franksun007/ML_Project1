# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""

    num_features = x.shape[1]
    counter = np.zeros(num_features)
    for i in range(len(x)):
        neg = np.where(x[i] == -999)
        counter[neg] += 1


    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x

    pos = np.where(counter > x.shape[0] * 0.5)

    for i in range(len(x)):
        x[i][pos] = 0

    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]

    # tx = np.hstack((np.ones((x.shape[0],1)), x))
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # split the data based on the given ratio:

    num_row = len(y)
    interval = int(num_row * ratio)
    indicies = np.random.permutation(num_row)
    x_train = x[indicies][: interval]
    y_train = y[indicies][: interval]
    x_test = x[indicies][interval:]
    y_test = y[indicies][interval:]
    return x_train, y_train, x_test, y_test


def performance(weights, y, xT):
    """Returns the percentage of successful classifications for the weights,
    given the expected results (y) and data (xT)"""
    from proj1_helpers import predict_labels
    compare_pred = predict_labels(weights, xT) - y
    return 1 - np.count_nonzero(compare_pred) / compare_pred.size
