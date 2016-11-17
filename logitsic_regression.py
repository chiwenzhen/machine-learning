# coding=utf-8
import numpy as np


def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


class LogisticRegression:
    coef = None
    iternum = 100
    alpha = 0.2

    def __init__(self):
        pass

    def fit(self, X, y):
        row_num, col_num = X.shape
        col_num += 1  # 增加无参数列
        X = np.c_[X, np.ones(row_num)]
        y = np.mat(y).reshape((row_num, 1))
        w = np.mat(np.ones(col_num)).reshape((col_num+1))

        for i in xrange(self.iternum):
            h = sigmoid(X * w)
            err = y - h
            w += self.alpha * X.transpose() * err
        self.w = w

    def predict(self, X):
        row_num, col_num = X.shape
        X = np.c_[X, np.ones(row_num)]
        y = sigmoid(X * self.w)
        y[y > 0.5] = 1
        y[y <= 0.5] = 0
        y = y.astype(int)
        return y


def samples(n_samples):
    x1 = np.random.uniform(-10.0, 10.0, n_samples)
    x2 = np.random.uniform(-10.0, 10.0, n_samples)
    y = x1 + x2 - 1
    y[y > 0] = 1
    y[y <= 0] = 0
    y = y.astype(int)
    X = np.array([x1, x2]).reshape(n_samples, 2)
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    n_samples = 100
    X_train, y_train = samples(n_samples)
    X_test, y_test = samples(20)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    count = 0
    for a, b in zip(y_test, y_pred):
        if a == b:
            count += 1
    print "accuracy=%f" % (1.0 * count / y_test.shape[0])


