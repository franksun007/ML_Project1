import numpy as np
from proj1_helpers import *

"""
Author: Group #23
            272785
            271691
            204988

This file reproduced the submitted version of the csv file on kaggle.
The loads test and train data and output a single csv file that represents the
results of our classification.

WARNING: the we ran 40000 iterations and divided the dataset into three part,
we trained using regularized logistic regression with gradient descent, which might
be a little bit slow.

The detailed description is in a separate file called README.md

"""

# Path of the training, test, and output files.
DATA_TRAIN_PATH = '../../data/train.csv'
DATA_TEST_PATH = '../../data/test.csv'
OUTPUT_PATH = '../../data/output.csv'

# load the training data
print("Loading training data")
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
print("Done")

# Constant to indicate +1 for classification
BINARY_CLASSIFICATION_1 = 1

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))


def calculate_loss_logistic_regression(y, tx, w):
    """compute the cost by negative log likelihood."""
    prediction = tx @ w

    y1 = np.where(y == BINARY_CLASSIFICATION_1)

    # Prevent loss to be inf or nan, so that if the prediction is
    # over 700, we keep the prediction as it is, instead of
    # taking the exponent of it.
    # As the result is only used to be an indication of the current
    # function, this approximation is considered as appropriate
    over_700 = np.where(prediction >= 700)

    prediction_result = np.log(1 + np.exp(prediction))
    prediction_result[over_700] = prediction[over_700]
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


def line_search_gamma(loss, loss_prev, gamma):
    """
    A function that will adjust the step size naively
    according to the previous loss function value and
    the current loss function value

    If the gamma is adjusted, it will be 2/3 or the
    original gamma value.
    """
    if (loss > loss_prev):
        gamma = gamma / 1.5
    return gamma


def logistic_regression_helper(y, tx, gamma, max_iters, lambda_):
    """
    Helper function that will perform the core logistic regression
    algorithm with ** Gradient Descent **.
    """
    w = np.zeros((tx.shape[1], 1))  # init guess for w
    threshold = 1e-8  # Threshold for converge
    loss_prev = 0  # the previous loss

    for iter in range(max_iters):
        # Get the loss of w.r.t to the current w and including the penalized term
        # lambda_ = 0 if performing pure logistic regression
        loss = calculate_loss_logistic_regression(y, tx, w) + lambda_ * np.linalg.norm(w, 2)
        # compute the gradient
        gradient = calculate_gradient_logistic_regression(y, tx, w)
        # update with step size
        w -= gradient * gamma

        # If converge
        if (loss_prev != 0) and np.abs(loss_prev - loss) < threshold:
            print("Reached Threshold, exit")
            break

        # Update gamma
        gamma = line_search_gamma(loss, loss_prev, gamma)
        loss_prev = loss
        if (iter % 100) == 0:
            print("Gamma: ", gamma)
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

    return w


"""
    Logistic Regression and Regularized Logistic Regression
    share the same core code. The only difference is that
    for logistic regression, lambda_, the regularization term
    is set to 0.
"""


# This function is not used
def logistic_regression(y, tx, gamma, max_iters):
    """ return the final w from the logistic regression """
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_=0)


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """ return the final w from the penalized logistic regression, with lambda_ as a non 0 value"""
    return logistic_regression_helper(y, tx, gamma, max_iters, lambda_)


def performance(weights, y, xT):
    """Returns the percentage of successful classifications for the weights,
    given the expected results (y) and data (xT)"""
    from proj1_helpers import predict_labels
    compare_pred = predict_labels(weights, xT)
    compare_pred -= y.reshape((len(y), 1))

    non_zero = 0
    for i in range(len(compare_pred)):
        if compare_pred[i] != 0:
            non_zero += 1

    return 1 - non_zero / compare_pred.size


def standardize_0123_helper(x):
    """
    Helper function that standardize the input data to mean 0 stddev 1.
    The function replace all the -999 entries with the mean of all non -999
    entries.
    """
    for i in range(x.shape[1]):
        mean = np.mean(x[np.where(x[:, i] != -999), i])
        x[np.where(x[:, i] == -999), i] = mean
        x[np.where(x[:, i] != -999), i] = x[np.where(x[:, i] != -999), i] - mean

    std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]

    return x


def standardize_0(x):
    """
    Standardize function for PRI_jet_num is 0
    Return a standardize version of the original feature, with
    useless features thrown away
    """
    # the features left that are meaningful and useful for training
    feature_left = np.array([0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    left_x = np.zeros((x.shape[0], len(feature_left)))
    left_x[:, :] = x[:, feature_left]
    return standardize_0123_helper(left_x)


def standardize_1(x):
    """
    Standardize function for PRI_jet_num is 1
    Return a standardize version of the original feature, with
    useless features thrown away
    """
    # the features left that are meaningful and useful for training
    feature_left = np.array([0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29])
    left_x = np.zeros((x.shape[0], len(feature_left)))
    left_x[:, :] = x[:, feature_left]
    return standardize_0123_helper(left_x)


def standardize_23(x):
    """
    Standardize function for PRI_jet_num is 2 or 3
    Return a standardize version of the original feature, with
    useless features thrown away
    """
    # the features left that are meaningful and useful for training
    feature_left = np.delete(np.arange(30), 22)
    left_x = np.zeros((x.shape[0], len(feature_left)))
    left_x[:, :] = x[:, feature_left]
    return standardize_0123_helper(left_x)


# The column index for PRI_jet_num
jet_num_col = 22


def split_dataset_wrt22(x):
    """
    Return three tuples of indices that splits x with respect to
    feature 22 - PRI_jet_num.
    First  Tuple of indicies: index in x where PRI_jet_num is 0
    Second Tuple of indicies: index in x where PRI_jet_num is 1
    Third  Tuple of indicies: index in x where PRI_jet_num is 2 or 3
    """
    x_22_0 = np.where(x[:, jet_num_col] == 0)
    x_22_1 = np.where(x[:, jet_num_col] == 1)
    x_22_23 = np.where(x[:, jet_num_col] >= 2)
    return x_22_0, x_22_1, x_22_23


def build_poly(x, degree):
    """
    Build the polynomial rising to the pass in parameter degree.
    Return a matrix that has the same entry as pass in x, while
    more features added according to degree.
    Each individual feature is a some power of the original feature.
    """
    matrix = np.zeros((x.shape[0], x.shape[1] * (degree + 1)))
    for i in range(degree + 1):
        matrix[:, (i * x.shape[1]): ((i + 1) * x.shape[1])] = (x ** i)[:]

    return matrix


def add_feature_helper(x, op, ori_shape):
    """
    Helper function that takes in x, an operator op, and the
    original shape of x.
    Return a matrix that is expanded with the feature added.
    The matrix will have the same entries as x, but additional
    ori_shape columns of feature added.
    """
    matrix = np.zeros((x.shape[0], x.shape[1] + ori_shape))
    matrix[:, : x.shape[1]] = x[:, :]
    matrix[:, x.shape[1] : ] = op(x[:, : ori_shape])
    return matrix


def add_feature(x):
    """
    Add some features that we consider as useful and meaningful
    to the data and good for training.
    Return a modified x with features added.
    """
    original_d = x.shape[1]
    x = add_feature_helper(x, np.sin, original_d)
    x = add_feature_helper(x, np.tanh, original_d)
#     x = add_feature_helper(x, np.cos, original_d)
    return x


# Max iteration
max_iter = 40000
lambdas = np.array([0.1])    # only a single lambda
gammas = np.array([0.005])   # only a single gamma
# polynomial degree
degree = 2

# split the data into 3 sets
i_0, i_1, i_23 = split_dataset_wrt22(tX)
tx_0 =  tX[i_0]
y_0 =   y[i_0]
tx_1 =  tX[i_1]
y_1 =   y[i_1]
tx_23 = tX[i_23]
y_23 =  y[i_23]

# Standardize the data
std_tx_0 = standardize_0(tx_0)
std_tx_1 = standardize_1(tx_1)
std_tx_23 = standardize_23(tx_23)

# Add the feature, not used
# std_tx_0 = add_feature(std_tx_0)
# std_tx_1 = add_feature(std_tx_1)
# std_tx_23 = add_feature(std_tx_23)

# Build the polynomial
matrix_std_tx_0 = build_poly(std_tx_0, degree)
matrix_std_tx_1 = build_poly(std_tx_1, degree)
matrix_std_tx_23 = build_poly(std_tx_23, degree)

# Perform Regularized logistic regression dataset where PRI_jet_num is 0
print("Training on feature 22 == 0")
weights_0 = reg_logistic_regression(y_0, matrix_std_tx_0, lambdas[0], gammas[0], max_iter)
print("Done")

# Perform Regularized logistic regression dataset where PRI_jet_num is 1
print("Training on feature 22 == 1")
weights_1 = reg_logistic_regression(y_1, matrix_std_tx_1, lambdas[0], gammas[0], max_iter)
print("Done")

# Perform Regularized logistic regression dataset where PRI_jet_num is 2 or 3
print("Training on feature 22 == 2 or 3")
weights_23 = reg_logistic_regression(y_23, matrix_std_tx_23, lambdas[0], gammas[0], max_iter)
print("Done")

# invoke the performance function to get a rough estimate on how well we are doing
# on the data that we just trained.
#
# We suppose to use cross-validation for this step.
# However, due to the characteristics of the data, and the selection of the model,
# we think that evaluate on the original training dataset will give us a reference on
# how well (bad) we are doing.
# This step is only an indication on whether we did anything REALLY wrong or not.
print("WARNING: THE PERFORMANCE SCORE IS ONLY USED TO INDICATE WHETHER YOU ARE VERY WRONG")
print("0  Size: ", len(y_0), "\tPerformance: ", performance(weights_0, y_0, matrix_std_tx_0))
print("1  Size: ", len(y_1), "\tPerformance: ", performance(weights_1, y_1, matrix_std_tx_1))
print("23 Size: ", len(y_23), "\tPerformance: ", performance(weights_23, y_23, matrix_std_tx_23))
print("")


# load test data
print("Loading test data...")
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
i_0_test, i_1_test, i_23_test = split_dataset_wrt22(tX_test)
print("Done")

# Pre-process the test data with the same method as training data

# split tx into 3 set
tx_0_test = tX_test[i_0_test]
tx_1_test = tX_test[i_1_test]
tx_23_test = tX_test[i_23_test]

# standardize
std_tx_0_test = standardize_0(tx_0_test)
std_tx_1_test = standardize_1(tx_1_test)
std_tx_23_test = standardize_23(tx_23_test)

# add feature, not used
# std_tx_0_test = add_feature(std_tx_0_test)
# std_tx_1_test = add_feature(std_tx_1_test)
# std_tx_23_test = add_feature(std_tx_23_test)

# split index into 3 features
ids_0_test = ids_test[i_0_test]
ids_1_test = ids_test[i_1_test]
ids_23_test = ids_test[i_23_test]

# Make prediction
y_pred_0 = predict_labels(weights_0, build_poly(std_tx_0_test, degree))
y_pred_1 = predict_labels(weights_1, build_poly(std_tx_1_test, degree))
y_pred_23 = predict_labels(weights_23, build_poly(std_tx_23_test, degree))

# concatenate everything into one
y_pred = np.concatenate((y_pred_0, y_pred_1, y_pred_23), axis=0)
ids_test = np.concatenate((ids_0_test, ids_1_test, ids_23_test), axis=0)

# Generate predictions and save output in csv format for submission:
print("Output to CSV...")
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print("Done")
