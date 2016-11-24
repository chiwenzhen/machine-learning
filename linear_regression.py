# coding=utf-8
import numpy as np
from descent import gradientdescent


# 线性回归
class LinearRegression:
    """Class for implimenting Linear Regression"""

    def __init__(self):
        """
        Attributes::
            learned (bool): Keeps track of if Linear Regression has been fit
            weights (np.ndarray): vector of weights for linear regression
        """
        self.learned = False
        self.weights = np.NaN

    def predict(self, X):
        """
        Args:
            X (np.ndarray): Test data of shape[n_samples, n_features]
        Returns:
            np.ndarray: shape[n_samples, 1], predicted values
        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        # Add column of 1s to X for perceptron threshold
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        prediction = np.dot(X, np.transpose(self.weights))
        return prediction

    def grad(self, X, y, weights):
        """
        Computes the gradient (needed if using gradient descent).

        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            weights (np.ndarray): Optional use of gradient descent to calculate weights
                if False, uses closed form solution to calculate weights.
        Returns:
            np.array: the gradient of the linear regression cost function
        """
        hypothesis = np.dot(X, weights) - y
        gradient = np.dot(np.transpose(X), hypothesis) / np.size(y)
        return gradient

    def fit(self, X, y, gradient=True, reg_parameter=0):
        """
        Currently, only L2 regularization is implemented.

        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            gradient (bool): Optional use of gradient descent to calculate weights
                if False, uses closed form solution to calculate weights.
            reg_parameter (float): float to determine strength of regulatrization  penalty
                if 0, then no linear regression without regularization is performed
        Returns: an instance of self
        """
        y = np.asarray(y)
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        if gradient:
            self.weights = gradientdescent(X, y, self.grad, reg_param=reg_parameter)
        else:
            # Calculate weights (closed form solution)
            XtX_lambaI = np.dot(np.transpose(X), X) + reg_parameter * np.identity(len(np.dot(np.transpose(X), X)))
            self.weights = np.dot(np.linalg.pinv(XtX_lambaI), np.dot(np.transpose(X), y))
        self.learned = True
        return self

