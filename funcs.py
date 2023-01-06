import numpy as np

def mse(x, y, weights):
    x = np.c_[x, np.ones(x.shape[0])]
    y_pred = x.dot(weights.T)
    return sum((y - y_pred)**2)/y.shape[0]


def mse_l1(x, y, weights):
    x = np.c_[x, np.ones(x.shape[0])]
    y_pred = x.dot(weights.T)
    return sum((y - y_pred)**2)/y.shape[0] + np.sum(np.abs(weights))

def mse_l2(x, y, weights):
    x = np.c_[x, np.ones(x.shape[0])]
    y_pred = x.dot(weights.T)
    return sum((y - y_pred)**2)/y.shape[0] + np.sum(weights**2)


def mae(x, y, weights):
    x = np.c_[x, np.ones(x.shape[0])]
    y_pred = x.dot(weights.T)
    return sum(np.abs(y - y_pred))/y.shape[0]

def cross_ent(y, num_classes):
    unique = list(set(y))
    y = list(y)
    res = []
    shape = len(y)
    for i in unique:
        res.append((y.count(i)/shape) * np.log2(y.count(i)/shape))
    return -sum(res)

def gini(y):
    unique, counts = np.unique(y, return_counts=True)
    counts = np.array(counts)
    counts = counts / y.shape[0]
    return 1 - np.sum(counts * counts)