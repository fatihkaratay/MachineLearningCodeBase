"""
This file contains functionality to add random and specified
data
"""

import numpy as np


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
