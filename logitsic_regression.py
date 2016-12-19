# coding=utf-8
import numpy as np
from descent import gradientdescent


# 逻辑回归
class LinearRegression:
    def __init__(self):
        self.weights = np.NaN

    def predict(self, X):
        # Add column of 1s to X for perceptron threshold
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        prediction = np.dot(X, np.transpose(self.weights))
        return prediction

    def grad(self, X, y, weights):
        hypothesis = np.dot(X, weights) - y
        gradient = np.dot(np.transpose(X), hypothesis) / np.size(y)
        return gradient

    def fit(self, X, y, reg_parameter=0):
        y = np.asarray(y)
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        self.weights = gradientdescent(X, y, self.grad, reg_param=reg_parameter)
        return self

