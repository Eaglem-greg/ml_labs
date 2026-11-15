import numpy as np
import pandas as pd



class MyBoostReg():
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5, min_samples_split=2, max_leafs=20, bins=16):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.pred_0 = None
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        self.pred_0 = np.mean(y)
        pred_m = np.array([self.pred_0 for i in range(len(X))])
        for _ in range(self.n_estimators):
            diff = y - pred_m 
            model = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            model.fit(X, diff)
            pred = model.predict(X)
            pred_m += self.learning_rate * np.array(pred)
            self.trees.append(model)
    
    def predict(self, X):
        if len(self.trees) == 0:
            raise ValueError('Градиентный бустинг не обучен')
        global_pred = self.pred_0
        for model in self.trees:
            pred = np.array(model.predict(X))
            global_pred += self.learning_rate * pred
        return global_pred
            
        
        
    def __str__(self):
        return f'MyBoostReg class: n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}'
    
    
    
class Node:
    def __init__(self, feature=None, threshold=None, gain=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.gain = gain
        self.left = left
        self.right = right

class Leaf:
    def __init__(self, samples=None, value=None):
        self.samples = samples
        self.value = value


class MyTreeReg():
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.tree = None
        self.bins = bins
        self.split_candidates = None
        self.fi = {}
        self._train_size = 0
    
    @staticmethod
    def get_best_split(X, y, split_candidates=None):
        names = X.columns

        def mse(target):
            y_mean = np.mean(target)
            return 1 / len(target) * np.sum((target - y_mean)**2)
       

        best_feature = None
        best_divider = None
        best_gain = float('-inf')
        total = len(y)
        mse0 = mse(y)

        for col in names:
            if split_candidates is None:
                vals = np.sort(X[col].unique())
                dividers = (vals[1:] + vals[:-1]) / 2
            else:
                dividers = split_candidates.get(col, [])
            
            for j in dividers:
                left_mask = X[col] <= j
                right_mask = X[col] > j

                left_count = left_mask.sum()
                right_count = right_mask.sum()

                if left_count == 0 or right_count == 0:
                    continue
                    
                left_mse = mse(y.loc[left_mask])
                right_mse = mse(y.loc[right_mask])
                
                gain = mse0 - (left_count/total * left_mse + right_count/total * right_mse)
                
                if gain > best_gain:
                    best_gain = gain
                    best_divider = j
                    best_feature = col

        return best_feature, best_divider, best_gain
    
    def _help_split_candidates(self, col):
        val = np.sort(col.unique())
        dividers = []
        for i in range(1, len(val)):
            mid = (val[i] + val[i-1]) / 2
            dividers.append(mid)
        if len(dividers) <= self.bins-1:
            self.split_candidates[col.name] = dividers
        else:
            counts, bin_edges = np.histogram(col.to_numpy(), self.bins)
            self.split_candidates[col.name] = bin_edges[1:-1]
            
    def _make_leaf(self, y):
        val = np.mean(y)
        samples = len(y)
        return Leaf(samples, val)

    def _build_tree(self, X, y, depth):
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or 
            y.nunique() == 1 or
            (depth > 0 and self.leafs_cnt >= self.max_leafs)
            ):
            return self._make_leaf(y)

        feature, threshold, gain = self.get_best_split(X, y, self.split_candidates)
        if feature is None or threshold is None:
            return self._make_leaf(y)

        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold
        X_left, y_left = X.loc[left_mask], y.loc[left_mask]
        X_right, y_right = X.loc[right_mask], y.loc[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            return self._make_leaf(y)

        self.leafs_cnt += 1
        
        node_weight = len(y) / self._train_size
        contribution = node_weight * gain
        self.fi[feature] += contribution

        left_child  = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return Node(feature, threshold, gain, left_child, right_child)


    def fit(self, X, y, samples=None):
        self.tree = None
        self.leafs_cnt = 1
        self.split_candidates = None
        for i in X.columns:
            self.fi[i] = 0
        if samples is None:
            self._train_size = len(X)
        else:
            self._train_size = samples
            
        if self.bins is not None:
            self.split_candidates = {}
            for i in X.columns:
                self._help_split_candidates(X[i])
        self.tree = self._build_tree(X, y, depth=0)

    def _print_node(self, node, depth):
        indent = '  ' * depth
        if isinstance(node, Leaf):
            print(indent + f'Класс: {node.value}, Количество: {node.samples}')
            return
        if isinstance(node, Node):
            print(indent + f'Фича: {node.feature}, Разделитель: {node.threshold}, Прирост информации: {node.ig}')
            print(indent + 'left: (<=)')
            self._print_node(node.left, depth+1)
            print(indent + 'right: (>)')
            self._print_node(node.right, depth+1)

    def print_tree(self):
        if self.tree is None:
            raise ValueError('Дерево не обучено')
        self._print_node(self.tree, 0)
    
    def _predict_row(self, row):
        cur = self.tree
        while isinstance(cur, Node):
            val = row[cur.feature]
            if val <= cur.threshold:
                cur = cur.left
            else:
                cur = cur.right
        return cur.value

    def predict(self, X):
        if self.tree is None:
            raise ValueError('Дерево не обучено')
        arr = []
        for _, row in X.iterrows():
            arr.append(self._predict_row(row))
        return arr


    def __str__(self):
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'