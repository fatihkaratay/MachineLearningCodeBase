"""
This file contains some functionalities that we use in the K-Means clustering
Algorithm
"""
import numpy as np


def find_closest_centroids(X, centroids):
    """
    Compute the centroid memberships for every example
    Args:
        X: input values
        centroids: k centroids

    Returns:
        idx: closest centroids
    """

    # set K
    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distance = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)

    return idx
