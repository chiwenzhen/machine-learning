# coding=utf-8
import numpy as np

if __name__ == "__main__":
    X = np.zeros((3, 2))
    X = np.c_[X, np.ones(2)]
    print X