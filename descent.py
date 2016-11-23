# coding=utf-8

import numpy as np


def gradientdescent(X, y, gradient, alpha=0.01, iterations=1000,
                    initial_weights=False, stochastic=False, reg_param=0):
    """
    Args:
        X (np.ndarray): 训练数据[n_samples, n_features]
        y (np.ndarray): 训练标记[n_samples, 1]
        gradient (function): 计算梯度的函数, 形式: gradient(X, y, weights)
        alpha (float): 调整梯度下降速度的参数
        iterations (int): 梯度下降次数
        initial_weights (np.ndarray): 初始权值
        stochastic (bool): 如果为True, 进行Stochastic Gradient Descent
        reg_param (float): 正则化参数, 0表示不进行正则化
    Returns:
        (np.ndarray): 权值[n_features, 1]
    """

    # If no initial weights given, initials weights = 0
    if not initial_weights:
        weights = np.zeros(np.shape(X)[1])
    else:
        weights = initial_weights

    # Update weights via gradient descent
    iteration = 0
    while iteration < iterations:
        if stochastic:
            random_index = np.random.randint(len(y))
            weights = weights - alpha * gradient(X[random_index], y[random_index], weights)
        else:
            weights = weights * (1 - alpha * reg_param / len(y)) - alpha * gradient(X, y, weights)
        iteration += 1
    return weights
