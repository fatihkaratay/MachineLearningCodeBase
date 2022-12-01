"""
This file contains functionality to add random and specified
data
"""

import numpy as np
from sklearn.model_selection import train_test_split

from supervised_learning.models.linear_regression import LinearRegressionModel


def generate_data(m, seed=1, scale=0.7):
    """
    Generates a data set based on a x^2 with added noise.
    """
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train ** 2 + c
    y_train = y_ideal + scale * y_ideal * (np.random.sample((m,)) - 0.5)
    x_ideal = x_train

    return x_train, y_train, x_ideal, y_ideal


def tune_m():
    """ tune the number of examples to reduce overfitting """
    m = 50
    m_range = np.array(m * np.arange(1, 16))
    num_steps = m_range.shape[0]
    degree = 16
    err_train = np.zeros(num_steps)
    err_cv = np.zeros(num_steps)
    y_pred = np.zeros((100, num_steps))

    for i in range(num_steps):
        X, y, y_ideal, x_ideal = generate_data(m_range[i], 5, 0.7)
        x = np.linspace(0, int(X.max()), 100)
        X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
        X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.50, random_state=1)

        lmodel = LinearRegressionModel(degree)  # no regularization
        lmodel.fit(X_train, y_train)
        yhat = lmodel.predict(X_train)
        err_train[i] = lmodel.mse(y_train, yhat)
        yhat = lmodel.predict(X_cv)
        err_cv[i] = lmodel.mse(y_cv, yhat)
        y_pred[:, i] = lmodel.predict(x)
    return (X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)
