# This is just a terminal version of the notebook


import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
DATA_TRAIN_PATH = '../../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

from logistic_regression import *
from helpers import *
from costs import *

# There are two parameters, lambda_ and gamma, where gamma is the step size

max_iter = 20000

lambdas = np.arange(0.1, 0.4, 0.1)

gammas = [0.5]

tx, _, std_x = standardize(tX)

weights = reg_logistic_regression(y, tx, lambdas[0], gammas[0], max_iter)
print(weights)

