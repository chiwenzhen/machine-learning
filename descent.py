# coding=utf-8

import numpy as np


def gradientdescent(X, y, gradient, alpha=0.01, iterations=100000, reg_param=0):
    weights = np.zeros(np.shape(X)[1])
    # Update weights via gradient descent
    iteration = 0
    while iteration < iterations:
        weights = weights * (1 - alpha * reg_param / len(y)) - alpha * gradient(X, y, weights)
        iteration += 1
    return weights
