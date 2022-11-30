"""
This file is the test file that runs against all supervised learning section
"""

import numpy as np


def test_eval_mse(target):
    y_hat = np.array([2.4, 4.2])
    y_tmp = np.array([2.3, 4.1])
    result = target(y_hat, y_tmp)

    assert np.isclose(result, 0.005, atol=1e-6), f"Wrong value. Expected 0.005, got {result}"

    y_hat = np.array([3.] * 10)
    y_tmp = np.array([3.] * 10)
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 0.), f"Wrong value. Expected 0.0 when y_hat == t_tmp, but got {result}"

    y_hat = np.array([3.])
    y_tmp = np.array([0.])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 4.5), f"Wrong value. Expected 4.5, but got {result}. Remember the square term"

    y_hat = np.array([3.] * 5)
    y_tmp = np.array([2.] * 5)
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 0.5), f"Wrong value. Expected 0.5, but got {result}. Remember to divide by (2*m)"

    print("\033[92m All tests passed.")
