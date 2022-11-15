"""
plotting the linear fit
"""

import matplotlib.pyplot as plt


def scatter_plot(x_train, y_train, title, x_label, y_label):
    plt.scatter(x_train, y_train, marker='x', c='r')

    # Set the title
    plt.title(title)
    # Set the y-axis label
    plt.ylabel(y_label)
    # Set the x-axis label
    plt.xlabel(x_label)
    plt.show()


def linear_fit_plot(x_train, y_train, predicted, title, x_label, y_label):
    # Plot the linear fit
    plt.plot(x_train, predicted, c="b")

    # Create a scatter plot of the data.
    plt.scatter(x_train, y_train, marker='x', c='r')

    # Set the title
    plt.title(title)
    # Set the y-axis label
    plt.ylabel(y_label)
    # Set the x-axis label
    plt.xlabel(x_label)
