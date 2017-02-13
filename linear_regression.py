# coding=utf-8
import numpy as np


# 线性回归
class LinearRegression:
    def __init__(self):
        self.weights = np.NaN

    # 预测
    def predict(self, X):
        # Add column of 1s to X for perceptron threshold
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        prediction = np.dot(X, np.transpose(self.weights))
        return prediction

    # 梯度
    def grad(self, X, y, weights):
        hypothesis = np.dot(X, weights)  # [n, 1]
        # loss = 1/2n * ∑(wx^i - y^i)**2
        gradient = np.dot(np.transpose(X), hypothesis - y) / np.size(y)  # values for all weight([n_features, 1])
        return gradient

    # 训练
    def fit(self, X, y):
        # prepare data
        y = np.asarray(y)
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))

        # update weights via gradient descent
        weights = np.zeros(np.shape(X)[1])
        reg_param = 0
        iteration = 0
        alpha = 0.01
        iterations = 100000
        reg_param = 0
        while iteration < iterations:
            weights = weights * (1 - alpha * reg_param / len(y)) - alpha * self.grad(X, y, weights)
            iteration += 1
        self.weights = weights
        return self

