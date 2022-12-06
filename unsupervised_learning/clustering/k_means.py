"""
This file contains some functionalities that we use in the K-Means clustering
Algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import *


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


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing
    the means of the data points assigned to each centroid

    Args:
        X: Data points
        idx: Array containing index of closest centroid for each example in X.
             Concretely, idx[i] contains the index of the centroid closest to example i.
        K: number of centroids

    Returns:
        new computed centroids array
    """
    m,n = X.shape
    centroids = np.zeros((K, n))
    for i in range(K):
        points = X[idx == i]
        centroids[i] = np.mean(points, axis=0)

    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print("K-Means iteration %d/%d" % (i, max_iters - 1))

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show()
    return centroids, idx
