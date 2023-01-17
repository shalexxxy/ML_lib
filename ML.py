import numpy as np
import pandas as pd
import funcs as f
from random import randrange

 # Linear Regression model (can be used 2 different ways of optimozation :
 # gradient descent for custom function or method of least squares with MSE loss func)
class LinearRegression:
    def __init__(self):
        pass

    def calc_grad(self, x, y, func, step=0.0001):
        # x - arguments, y - target, func - loss, step - gradient step
        grad = np.array([0 for _ in range(self.weights.shape[0])])
        for i in range(len(self.weights)):
            weights = self.weights.copy() # Defining copy of weights vector
            weights[i] += step # adding step
            grad[i] = (func(x, y, weights) - func(x, y, self.weights)) / step # calc grad
        return grad
# fitting thw model for method of least squares
    def fit(self, x_train, y_train):
        x_train = np.c_[x_train, np.ones(x_train.shape[0])] # adding free term
        a = x_train.T.dot(x_train)
        b = x_train.T.dot(y_train)
        self.weights = np.linalg.solve(a, b) # solving the equation
 # fitting model with custom loss
    def fit_custom(self, x, y, learning_rate, func=f.mse):
        self.weights = np.random.normal(0, 1, len(x[0]) + 1)
        loss = np.inf
        current_loss = func(x, y, self.weights)
        # calculating grad steps until difference between loss funcs will be < 10**8 %
        while (abs(loss - current_loss) / current_loss) > 10 ** (-10):
            loss = current_loss
            grad = self.calc_grad(x, y, func)
            self.weights = self.weights - grad * learning_rate # adding anti-grad with lr
            #   print(self.weights)
            current_loss = func(x, y, self.weights)

    def predict(self, x):
        x = np.c_[x, np.ones(x.shape[0])]
        return x.dot(self.weights.T)

# tree node class
class Node:
    def __init__(self):
        self.split_val = None # value of the feature_num to split
        self.depth = None # depth of the current node
        self.feature_num = None # number of the feature to split
        self.left = None
        self.right = None
        self.mass = None # mass of target values in the node


class ClassificationTree:
    def __init__(self):
        self.root = Node()
        self.root.depth = 1

# Recursive func for split
    def split_current(self, data, current_node, min_elems, max_depth):
        x = data.loc[:, data.columns != 'labels']
        cols = x.columns
        min_gini_split = np.inf
        min_col = 1
        min_val = - np.inf
        # for each unique value of the feature calc gini and tring to find the best split
        for i in cols:
            for j in list(sorted(data[i].unique())):
                metric = (data[data[i] >= j].shape[0] / data.shape[0]) * f.gini(data[data[i] >= j]['labels']) + \
                         (data[data[i] < j].shape[0] / data.shape[0]) * f.gini(data[data[i] < j]['labels'])
                if metric < min_gini_split:
                    min_gini_split = metric
                    min_col = i
                    min_val = j
        # creating new Nodes for split
        current_node.right = Node()
        current_node.left = Node()
        current_node.right.depth = current_node.depth + 1
        current_node.left.depth = current_node.depth + 1
        current_node.left.mass = data[data[min_col] < min_val]
        current_node.right.mass = data[data[min_col] >= min_val]
        current_node.feature_num = min_col
        current_node.split_val = min_val
        # if new node satisfies requirements - we continue splitting
        if (current_node.right.mass.shape[0] > min_elems) and (current_node.right.mass['labels'].nunique() > 1) and (current_node.right.depth < max_depth):
            self.split_current(current_node.right.mass, current_node.right, min_elems, max_depth)
        if (current_node.left.mass.shape[0] > min_elems) and (current_node.left.mass['labels'].nunique() > 1) and (current_node.left.depth < max_depth):
            self.split_current(current_node.left.mass, current_node.left, min_elems, max_depth)
    # fit func
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
        for i in x.index: # itarating rows of features
            current_node = self.root
            while current_node is not None: # walking until reaching the list
                if current_node.right is None or current_node.left is None:
                    mass = current_node.mass['labels']
                    current_node = None
                else:
                    if x.loc[i, current_node.feature_num] < current_node.split_val and current_node.left is not None:
                        current_node = current_node.left
                    elif x.loc[i, current_node.feature_num] >= current_node.split_val and current_node.right is not None:
                        current_node = current_node.right
           # print(mass)
            # as a prediction we choose the most frequent label
            res.append(mass.value_counts().idxmax())
        return res


## Same as classification tree
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
        return res

# Making and fitting num_trees of RegTrees, the prediction is mean value of predictions
class RandomRegressionForest:
    def __init__(self, max_depth = 5, min_leaf_elems = 5, num_trees = 10):
        self.max_depth = max_depth
        self.min_leaf_elems = min_leaf_elems
        self.num_trees = num_trees
        self.forest = [RegressionTree() for _ in range(num_trees)]

    def fit(self, x_train, y_train):
        self.parms = []
        data = pd.DataFrame(x_train)
        num_samples = data.shape[0]//self.num_trees
        data['target'] = y_train
        x = None
        for i in range(len(self.forest)):
            data_train = data.sample(num_samples)
            target = data_train.pop('target')
            x = data_train.sample(randrange(1,len(data_train.columns)), axis=1)
            self.parms.append(x.columns)
            self.forest[i].fit(x, target, self.min_leaf_elems, self.max_depth)

    def predict(self, x):
        self.predictions = []
        x = pd.DataFrame(x)
        for i in range(len(self.forest)):
            self.predictions.append(np.array (self.forest[i].predict(x[list(self.parms[i]) ])))
        res = sum(self.predictions)/self.num_trees
        return res

class RandomClassificationForest:
    def __init__(self, max_depth=5, min_leaf_elems=10, num_trees = 5):
        self.max_depth = max_depth
        self.min_leaf_elems = min_leaf_elems
        self.num_trees = num_trees
        self.forest = [ClassificationTree() for _ in range(num_trees)]

    def fit(self, x_train, y_train):
        self.parms = []
        data = pd.DataFrame(x_train)
        num_samples = data.shape[0]//self.num_trees
        data['labels'] = y_train
        for i in range(len(self.forest)):
            data_train = data.sample(num_samples)
            target = data_train.pop('labels')
            x = data_train.sample(randrange(1,len(data_train.columns)), axis=1)
            self.parms.append(x.columns)
            self.forest[i].fit(x, target, self.min_leaf_elems, self.max_depth)
        print('model was fitted')

    def predict(self, x):
        self.predictions = []
        x = pd.DataFrame(x)
        for i in range(len(self.forest)):
            self.predictions.append(np.array (self.forest[i].predict(x[list(self.parms[i]) ])))
            print(i, ' predicted')
        self.predictions  = pd.DataFrame(self.predictions).T
        res = []
        print(self.predictions)
        for i in list(self.predictions.index):
            val = self.predictions.loc[i, :].value_counts().idxmax()
            res.append(val)
            print(val)
        print(res)
        return res

