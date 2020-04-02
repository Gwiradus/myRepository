import random

import numpy as np


class MyExtraTreesRegression:

    def __init__(self):
        self.tree = None

    def fit(self, x, y, ktr, n_min_leaf=2):
        self.tree = Node(x, y, ktr, np.array(np.arange(len(y))), n_min_leaf)
        return self

    def predict(self, x):
        return self.tree.predict(x.values)


class Node:

    def __init__(self, x, y, ktr, idxs, n_min_leaf=2):
        self.x = x
        self.y = y
        self.ktr = ktr
        self.index = idxs
        self.n_min_leaf = n_min_leaf
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('-inf')
        self.split_node(ktr)

    def split_node(self, ktr):
        samples = self.x.values[self.index].shape[0]
        if samples < self.n_min_leaf:  # If node contains less than n_min_leaf samples
            return
        for c in range(ktr):
            # self.random_split(random.randint(0, self.x.shape[1]-1))
            self.random_split(c)
        if self.is_leaf:  # If node is leaf
            return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(self.x, self.y, self.ktr, self.index[lhs], self.n_min_leaf)
        self.rhs = Node(self.x, self.y, self.ktr, self.index[rhs], self.n_min_leaf)

    def random_split(self, var_idx):
        x = self.x.values[self.index, var_idx]
        x_min = np.amin(x)
        x_max = np.amax(x)
        x_split = random.uniform(x_min, x_max)
        lhs = x <= x_split
        rhs = x > x_split
        if rhs.sum() < 1 or lhs.sum() < 1:  # If all attributes are constant in x
            return
        curr_score = self.calc_score(lhs, rhs)
        if curr_score > self.score:
            self.var_idx = var_idx
            self.score = curr_score
            self.split = x_split

    def calc_score(self, lhs, rhs):
        y = self.y[self.index]
        x_var = np.var(y)
        if x_var == 0:  # If the output is constant in x
            return float('-inf')
        x_card = y.shape[0]
        lhs_var = np.var(y[lhs])
        lhs_card = y[lhs].shape[0]
        rhs_var = np.var(y[rhs])
        rhs_card = y[rhs].shape[0]
        calc = (x_var - ((lhs_card / x_card) * lhs_var) - ((rhs_card / x_card) * rhs_var)) / x_var
        return calc

    @property
    def split_col(self):
        return self.x.values[self.index, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('-inf')

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)
