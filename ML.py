import numpy as np
import pandas as pd
import funcs as f

class LinearRegression:
    def __init__(self, reg=None):
        self.reg = reg

    def calc_grad(self, x, y, func, step=0.0001):
        grad = np.array([0 for i in range(self.weights.shape[0])])

        for i in range(len(self.weights)):
            weights = self.weights.copy()
            weights[i] += step
            grad[i] = (func(x, y, weights) - func(x, y, self.weights)) / step
        return grad

    def fit(self, x_train, y_train):
        x_train = np.c_[x_train, np.ones(x_train.shape[0])]
        a = x_train.T.dot(x_train)
        b = x_train.T.dot(y_train)
        self.weights = np.linalg.solve(a, b)

    def fit_custom(self, x, y, learning_rate, func=f.mse):
        self.weights = np.random.normal(0, 1, len(x[0]) + 1)
        loss = np.inf
        current_loss = func(x, y, self.weights)
        while (abs(loss - current_loss) / current_loss) > 10 ** (-10):
            loss = current_loss
            grad = self.calc_grad(x, y, func)
            self.weights = self.weights - grad * learning_rate
            #   print(self.weights)
            current_loss = func(x, y, self.weights)

    def predict(self, x):
        x = np.c_[x, np.ones(x.shape[0])]
        return x.dot(self.weights.T)


class Node:
    def __init__(self):
        self.split_val = None
        self.depth = None
        self.feature_num = None
        self.left = None
        self.right = None
        self.mass = None

class ClassificationTree:
    def __init__(self):
        self.root = Node()
        self.root.depth = 1


    def split_current(self, data, current_node, min_elems, max_depth):
        x = data.loc[:, data.columns != 'labels']
        cols = x.columns
        min_gini_split = np.inf
        min_col = 1
        min_val = - np.inf
        for i in cols:
            for j in list(sorted(data[i].unique())):
                metric = (data[data[i] >= j].shape[0] / data.shape[0]) * f.gini(data[data[i] >= j]['labels']) + \
                         (data[data[i] < j].shape[0] / data.shape[0]) * f.gini(data[data[i] < j]['labels'])
                if metric < min_gini_split:
                    min_gini_split = metric
                    min_col = i
                    min_val = j

        current_node.right = Node()
        current_node.left = Node()
        current_node.right.depth = current_node.depth + 1
        current_node.left.depth = current_node.depth + 1
        current_node.left.mass = data[data[min_col] < min_val]
        current_node.right.mass = data[data[min_col] >= min_val]
        current_node.feature_num = min_col
        current_node.split_val = min_val
        print('Current depth',  current_node.right.depth)
        print('Current split ', min_gini_split)
        print(min_col, min_val)
        if (current_node.right.mass.shape[0] > min_elems) and (current_node.right.mass['labels'].nunique() > 1) and (current_node.right.depth < max_depth):
            self.split_current(current_node.right.mass, current_node.right, min_elems, max_depth)
        if (current_node.left.mass.shape[0] > min_elems) and (current_node.left.mass['labels'].nunique() > 1) and (current_node.left.depth < max_depth):
            self.split_current(current_node.left.mass, current_node.left, min_elems, max_depth)

    def fit(self, x_train, y_train, min_leaf_elems = 10, max_depth = 10):
        data = pd.DataFrame(x_train)
        data['labels'] = y_train
        depth = 1
        current_node = self.root
        current_node.mass = data
        self.split_current(current_node.mass, current_node, min_leaf_elems, max_depth)

    def predict(self, x):
        x = pd.DataFrame(x)
        res = []
        for i in x.index:
            current_node = self.root
            while current_node is not None:
                if current_node.right is None or current_node.left is None:
                    mass = current_node.mass['labels']
                    current_node = None
                else:
                    if x.loc[i, current_node.feature_num] < current_node.split_val and current_node.left is not None:
                        current_node = current_node.left
                    elif x.loc[i, current_node.feature_num] >= current_node.split_val and current_node.right is not None:
                        current_node = current_node.right

            res.append(mass.value_counts().idxmax())
        return res



class RegressionTree:
    def __init__(self):
        self.root = Node()
        self.root.depth = 1


    def split_current(self, data, current_node, min_elems, max_depth):
        x = data.loc[:, data.columns != 'target']
        cols = x.columns
        min_rss = np.inf
        min_col = 1
        min_val = - np.inf
        for i in cols:
            for j in list(sorted(data[i].unique())):
                rss = np.sum(np.abs(data[data[i] >= j]['target'] - np.mean(data[data[i] >= j]['target']))) + \
                np.sum(np.abs(data[data[i] < j]['target'] - np.mean(data[data[i] < j]['target'])))
                if rss < min_rss:
                    min_rss = rss
                    min_col = i
                    min_val = j

        current_node.right = Node()
        current_node.left = Node()
        current_node.right.depth = current_node.depth + 1
        current_node.left.depth = current_node.depth + 1
        current_node.left.mass = data[data[min_col] < min_val]
        current_node.right.mass = data[data[min_col] >= min_val]
        current_node.feature_num = min_col
        current_node.split_val = min_val
        #print('Current depth',  current_node.right.depth)
        #print('Current split ', min_rss)
        #print(min_col, min_val)
        if (current_node.right.mass.shape[0] > min_elems) and (current_node.right.mass['target'].nunique() > 1) and (current_node.right.depth < max_depth):
            self.split_current(current_node.right.mass, current_node.right, min_elems, max_depth)
        if (current_node.left.mass.shape[0] > min_elems) and (current_node.left.mass['target'].nunique() > 1) and (current_node.left.depth < max_depth):
            self.split_current(current_node.left.mass, current_node.left, min_elems, max_depth)

    def fit(self, x_train, y_train, min_leaf_elems = 10, max_depth = 10):
        data = pd.DataFrame(x_train)
        data['target'] = y_train
        depth = 1
        current_node = self.root
        current_node.mass = data
        self.split_current(current_node.mass, current_node, min_leaf_elems, max_depth)

    def predict(self, x):
        x = pd.DataFrame(x)
        res = []
        for i in x.index:
            current_node = self.root
            while current_node is not None:
                if current_node.right is None or current_node.left is None:
                    mass = current_node.mass['target']
                    current_node = None
                else:
                    if x.loc[i, current_node.feature_num] < current_node.split_val and current_node.left is not None:
                        current_node = current_node.left
                    elif x.loc[i, current_node.feature_num] >= current_node.split_val and current_node.right is not None:
                        current_node = current_node.right

            res.append(np.mean(mass))
            if np.mean(mass) is None:

                print('None is: ', mass)
        return res
