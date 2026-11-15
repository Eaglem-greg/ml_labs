import numpy as np
import pandas as pd

class MyForestClf():
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42, max_depth=5, min_samples_split = 2, max_leafs = 20, bins=16, criterion='entropy', oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion
        self.leafs_cnt = 0
        self.trees = []
        self.fi = {}
        self.oob_score = oob_score
        self.oob_score_ = None
        
    @staticmethod
    def metrics(metric, y_pred, y_class, y_true_class):
        if metric is None:
            return None
        tp = np.sum((y_class==1) & (y_true_class==1))
        tn = np.sum((y_class==0) & (y_true_class==0))
        fp = np.sum((y_class==1) & (y_true_class==0))
        fn = np.sum((y_class==0) & (y_true_class==1))
        if metric == 'accuracy':
            return (tp + tn) / (tp + tn + fp + fn)
        if metric == 'precision':
            return tp / (tp + fp)
        if metric == 'recall':
            return tp / (tp + fn)
        if metric == 'f1':
            precision = MyForestClf.metrics('precision', y_pred, y_class, y_true_class)
            recall = MyForestClf.metrics('recall', y_pred, y_class, y_true_class)
            return 2 * precision * recall / (precision + recall)
        if metric == 'roc_auc':
            y_pred = np.round(y_pred, 10)
            pos = y_pred[y_true_class == 1]
            neg = y_pred[y_true_class == 0]
            p = len(pos)
            n = len(neg)
            if p == 0 or n == 0:
                return 0.0
            diff = neg[:, None] - pos[None, :]
            wins = (diff < 0).astype(float) + 0.5 * (diff == 0)
            return wins.sum() / (p * n)
        
        
    def fit(self, X, y):
        random.seed(self.random_state)
        self.trees = []
        self.leafs_cnt = 0
        oob_arr = [[] for i in range(len(X))]
        for i in X.columns:
            self.fi[i] = 0
        for i in range(self.n_estimators):
            samples_count = round(X.shape[0] * self.max_samples)
            features_count = round(X.shape[1] * self.max_features)
            cols_idx = random.sample(list(X.columns), features_count)
            rows_idx = random.sample(range(X.shape[0]), samples_count)
            
            X_sample = X.iloc[rows_idx][cols_idx]
            y_sample = y.iloc[rows_idx]
            
            tree = MyTreeClf(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, self.criterion)
            tree.fit(X_sample, y_sample, len(X))
            for features, importance in tree.fi.items():
                self.fi[features] += importance
            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt
            
            oob_index = [i for i in range(len(X)) if i not in set(rows_idx)]
            if self.oob_score is not None and len(oob_index) > 0:
                oob_X = X.iloc[oob_index][cols_idx]
                oob_Y = tree.predict_proba(oob_X)
                for poz, pred in zip(oob_index, oob_Y):
                    oob_arr[poz].append(pred)
                    
        valid_indices = [i for i, predictions in enumerate(oob_arr) if len(predictions) > 0]
        y_true = y.iloc[valid_indices].to_numpy()
        y_pred = np.array([np.mean(oob_arr[i]) for i in valid_indices])
        y_pred_classes = np.array([1 if i > 0.5 else 0 for i in y_pred])
        self.oob_score_ = self.metrics(self.oob_score, y_pred, y_pred_classes, y_true)
            
    def predict(self, X, type='mean'):
        if len(self.trees) == 0:
            raise ValueError('Лес не обучен')
        res = []
        if type == 'mean':
            for _, row in X.iterrows():
                pred = 0
                for tree in self.trees:
                        pred += tree._predict_row(row)
                mean_pred = pred / len(self.trees)
                res.append(1 if mean_pred > 0.5 else 0)
        if type == 'vote':
            for _, row in X.iterrows():
                pred = []
                for tree in self.trees:
                        pred.append(1 if tree._predict_row(row) > 0.5 else 0)
                moda_pred = pd.Series(pred).mode()[0]
                res.append(moda_pred)
            
        return np.array(res)
    
    def predict_proba(self, X):
        if len(self.trees) == 0:
            raise ValueError('Лес не обучен')
        res = []
        for _, row in X.iterrows():
            pred = 0
            for tree in self.trees:
                pred += tree._predict_row(row)
            mean_pred = pred / len(self.trees)
            res.append(mean_pred)
        return np.array(res)
    
    def __str__(self):
        return f'MyForestClf class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, criterion={self.criterion}, random_state={self.random_state}'
    

class Node:
    def __init__(self, feature=None, threshold=None, ig=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.ig = ig
        self.left = left
        self.right = right

class Leaf:
    def __init__(self, samples=None, value=None):
        self.samples = samples
        self.value = value


class MyTreeClf():
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.max_leafs = max_leafs 
        self.leafs_cnt = 0
        self.tree = None
        self.bins = bins
        self.split_candidates = None
        self.criterion = criterion
        self.fi = {}
        self._train_size = 0
    
    @staticmethod
    def get_best_split(X, y, split_candidates=None, criterion='entropy'):
        names = X.columns

        def entropy(target):
            arr_probabilities = target.value_counts(normalize=True).to_numpy()
            return -np.sum(arr_probabilities * np.log2(arr_probabilities))
        
        def gini(target):
            arr_probabilities = target.value_counts(normalize=True).to_numpy()
            return 1 - np.sum(arr_probabilities**2)

        best_feature = None
        best_divider = None
        best_ig = float('-inf')
        total = len(y)
        if criterion == 'entropy':
            s0 = entropy(y) 
        if criterion == 'gini': 
            g0 = gini(y)

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

                if criterion == 'entropy':
                    left_s = entropy(y.loc[left_mask])
                    right_s = entropy(y.loc[right_mask])
                    ig = s0 - (left_count/total * left_s + right_count/total * right_s)
                if criterion == 'gini':
                    g_l = gini(y.loc[left_mask])
                    g_r = gini(y.loc[right_mask])
                    ig = g0 - (left_count/total * g_l + right_count/total * g_r)

                if ig > best_ig:
                    best_ig = ig
                    best_divider = j
                    best_feature = col

        return best_feature, best_divider, best_ig

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
        moda = np.mean(y)
        samples = len(y)
        return Leaf(samples, moda)

    def _build_tree(self, X, y, depth):
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or 
            y.nunique() == 1 or
            (depth > 0 and self.leafs_cnt >= self.max_leafs)
            ):
            return self._make_leaf(y)

        feature, threshold, ig = self.get_best_split(X, y, self.split_candidates, self.criterion)
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
        contribution = node_weight * ig
        self.fi[feature] += contribution
        left_child  = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return Node(feature, threshold, ig, left_child, right_child)


    def fit(self, X, y, samples):
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

    def predict_proba(self, X):
        if self.tree is None:
            raise ValueError('Дерево не обучено')
        arr = []
        for _, row in X.iterrows():
            arr.append(self._predict_row(row))
        return arr
    
    def predict(self, X): 
        arr = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in arr]

    def __str__(self):
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'