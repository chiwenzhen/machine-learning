# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import random
from linear_regression import LinearRegression


# 拟合目标函数
def f(x):
    return 2 * x + 1


# 生成数据
def generate(n_samples=100):
    X = (np.random.rand(n_samples) * 10 - 5)
    y = f(X) + np.random.normal(0.0, 10, n_samples)
    return X, y


if __name__ == '__main__':

    # 创建模型
    regr = LinearRegression()

    # 生成训练和测试数据
    x_train, y_train = generate(100)
    x_test, y_test = generate(50)

    # 训练模型
    regr.fit(x_train, y_train)

    # 测试模型
    y_pred = regr.predict(x_test)

    # 均方误差
    print("Mean squared error: %.2f" % (np.mean(y_test - y_test) ** 2))

    # 绘图
    plt.scatter(x_test, y_pred,  color='red')
    plt.plot(x_test, f(x_test), color='blue', linewidth=3)
    plt.show()