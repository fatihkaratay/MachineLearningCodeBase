"""
This file contains error functions that we use for the
error evaluation in supervised learning.
"""


def eval_mse(y, yhat):
    """
    Calculate the mean squared error on a dataset
    Args:
        y: ndarray, target value of each example
        yhat: ndarray, predicted value of each example

    Returns:
        err: scalar
    """
    m = len(y)
    err = 0.0
    for i in range(m):
        err_i = ((yhat[i] - y[i]) ** 2)
        err += err_i
    err = err / (2 * m)

    return err
