import numpy as np
import pandas as pd
import random

from types import FunctionType
class MyBoostClf():
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5, min_samples_split = 2, max_leafs=20,
                 bins = 16, metric=None, max_features=0.5, max_samples=0.5, random_state=42, reg=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.pred_0 = None
        self.trees = []
        self.metric = metric
        self.best_score = None
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.fi = {}
        
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
            precision = MyBoostClf.metrics('precision', y_pred, y_class, y_true_class)
            recall = MyBoostClf.metrics('recall', y_pred, y_class, y_true_class)
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
        
    def fit(self, X, y, X_eval=None, y_eval=None, early_stopping=None, verbose=None):
        random.seed(self.random_state)
        self.trees = []
        self.fi = {col: 0.0 for col in X.columns}
        p = np.mean(y)
        self.pred_0 = np.log(p / (1 - p))
        pred_m = np.array([self.pred_0 for i in range(len(X))])
        count_leafs = 0 
        
        cond = (X_eval is not None and y_eval is not None and early_stopping is not None)

        best_eval_score = None
        cnt_bad = 0

        if cond:
            eval_pred = np.full(len(X_eval), self.pred_0, dtype=float)
            
        for i in range(self.n_estimators):
            count_rows = round(len(X) * self.max_samples)
            count_cols = round(len(X.columns) * self.max_features)
            if count_rows == 0 or count_cols == 0:
                sample_x = X
                sample_y = y
                sample_pred = pred_m
            else:
                cols_idx = random.sample(list(X.columns), count_cols)
                rows_idx = random.sample(range(len(X)), count_rows)
            
                sample_x = X.iloc[rows_idx].loc[:, cols_idx]
                sample_y = y.iloc[rows_idx]
                sample_pred = pred_m[rows_idx]
                               
            p = np.exp(sample_pred) / (1 + np.exp(sample_pred))
            r = pd.Series(sample_y - p)
                               
            model = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            model.fit(sample_x, r, len(X))
            
            model._change_leafs(model.tree, sample_x, r, p, count_leafs, self.reg)
            self.trees.append(model)
            count_leafs += model.leafs_cnt
            
            for col, fi in model.fi.items():
                self.fi[col] += fi
                
            pred = model.predict(X)
            
            if isinstance(self.learning_rate, FunctionType):
                cur_lr = self.learning_rate(i + 1)
            else:
                cur_lr = self.learning_rate
                
            pred_m += cur_lr * np.array(pred)
            
            if cond:
                eval_pred_x = np.array(model.predict(X_eval))
                eval_pred += cur_lr * eval_pred_x
            
            p_pred = np.exp(pred_m) / (1 + np.exp(pred_m))
            LogLoss = -np.mean(y * np.log(p_pred + 10**(-15)) + (1 - y) * np.log(1 - p_pred + 10**(-15)))
            
            if cond:
                eval_proba = np.exp(eval_pred) / (1 + np.exp(eval_pred))
                eval_pred_classes = (eval_proba > 0.5).astype(int)
                if self.metric is not None:
                    eval_score = MyBoostClf.metrics(self.metric, eval_proba, eval_pred_classes, y_eval)
                else:
                    eval_score = -np.mean(y_eval * np.log(eval_proba + 10**(-15)) + (1 - y_eval) * np.log(1 - eval_proba + 10**(-15)))
                
                if best_eval_score is None:
                    best_eval_score = eval_score
                    cnt_bad = 0
                else:
                    if self.metric is None:
                        improved = eval_score < best_eval_score
                    else:
                        improved = eval_score > best_eval_score

                    if improved:
                        best_eval_score = eval_score
                        cnt_bad = 0
                    else:
                        cnt_bad += 1

                self.best_score = best_eval_score
                
            if self.metric is not None:
                metric_value = MyBoostClf.metrics(self.metric, p_pred, (p_pred > 0.5).astype(int), y) if not cond else eval_score
                    
                if verbose is not None:
                    if i % verbose == 0:
                        print(f'{i}. {LogLoss} | {self.metric}: {metric_value}')
            else:
                LogLoss = LogLoss if not cond else eval_score

                if verbose:
                    if i % verbose == 0:
                        print(f'{i}. {LogLoss}')
                          
                            
            if cond and cnt_bad >= early_stopping:
                del self.trees[-early_stopping:]
                
                self.fi = {col: 0.0 for col in X.columns}
                
                for model in self.trees:
                    for col, fi in model.fi.items():
                        self.fi[col] += fi 
                break
                    
    def predict_proba(self, X):
        if len(self.trees) == 0:
            raise ValueError('Градиентный бустинг не обучен')
        pred_m = np.array([self.pred_0 for i in range(len(X))])
        for i, model in enumerate(self.trees):
            if isinstance(self.learning_rate, FunctionType):
                pred_m += self.learning_rate(i+1) * np.array(model.predict(X))
            else:
                pred_m += self.learning_rate * np.array(model.predict(X))
            
        predictions = np.exp(pred_m) / (1 + np.exp(pred_m))
        
        return predictions
            
    
    def predict(self, X):
        predictions = self.predict_proba(X)
        return (predictions > 0.5).astype(int)
            
            
        
    def __str__(self):
        return f'MyBoostClf class: n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}'
    
    
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
        
        
    def _change_leafs(self, node, x, r, p, count_leafs, reg):
        if isinstance(node, Node):
            mask_left = x[node.feature] <= node.threshold
            mask_right = x[node.feature] > node.threshold
            left = x.loc[mask_left]
            right = x.loc[mask_right]
            r_left = r.loc[mask_left]
            r_right = r.loc[mask_right]
            p_left = p[mask_left]
            p_right = p[mask_right]
            self._change_leafs(node.left, left, r_left, p_left, count_leafs, reg)
            self._change_leafs(node.right, right, r_right, p_right, count_leafs, reg)
        if isinstance(node, Leaf):
            node.value = np.sum(r) / np.sum(p*(1 - p)) + count_leafs * reg
                
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