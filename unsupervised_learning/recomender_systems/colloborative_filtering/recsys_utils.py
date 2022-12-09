import numpy as np
import pandas as pd
from numpy import loadtxt


def normalize_ratings(Y, R):
    """
    Preprocess data by substracking mean rating for every movie (every row)
    Only include real ratings R(i, j)=1
    [Ynorm, Ymean]=normalize_ratings(Y, R) normalized Y so that each movie has
    a rating of 0 on average. Unrated movies then have a mean rating (0)
    Returns the mean rating in Ymean
    """
    Ymean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1))).reshape(-1, 1)
    Ynorm = Y - np.multiply(Ymean, R)
    return (Ynorm, Ymean)

