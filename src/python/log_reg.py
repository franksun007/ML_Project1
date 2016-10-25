import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
DATA_TRAIN_PATH = '../../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

from logistic_regression import *
from costs import *

# There are two parameters, lambda_ and gamma, where gamma is the step size

max_iter = 15000
lambdas = np.arange(0, 0.4, 0.1)
gammas = np.arange(0.01, 0.1, 0.01)

# When lambda is 0, reg_logistic_regression is naive non-penalized logistic_regression.

max_iter = 20000
lambdas = np.arange(0.1, 0.4, 0.1)
gammas = np.arange(0.001, 0.01, 0.001)

# When lambda is 0, reg_logistic_regression is naive non-penalized logistic_regression.


weights = reg_logistic_regression(y, tX, lambdas[0], gammas[0], max_iter)
# err = compute_loss(y, tX, w)
# struct[(gamma, lambda_)] = (w, err)
# #         break
# for (gamma, lambda_), (w, err) in struct.items():
#     print("Gamma: ", gamma, " Lamdba: ", lambda_, " w: ", w, "error: ", err)

