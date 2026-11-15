import numpy as np
import pandas as pd
import random



from types import FunctionType
class MyBoostReg():
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, loss='MSE', metric=None, max_features=0.5, max_samples=0.5, random_state=42, reg=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.pred_0 = None
        self.trees = []
        self.loss = loss
        self.metric = metric
        self.best_score = None
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.fi = {}
        
    @staticmethod
    def metrics(type_metric, y_pred, y, samples):
        if type_metric is None:
            return None
        if type_metric == 'MAE':
            return (1/samples) * np.sum(abs(y_pred - y))
        if type_metric == 'MSE':
            return (1/samples) * np.sum((y_pred - y)**2)
        if type_metric == 'RMSE':
            return np.sqrt((1/samples) * np.sum((y_pred - y)**2))
        if type_metric == 'MAPE':
            return (100/samples) * np.sum(abs((y_pred - y)/y))
        if type_metric == 'R2':
            up = np.sum((y_pred - y)**2)
            low = np.sum((y - np.mean(y))**2)
            if low == 0:
                return 1
            return 1 - up / low
        
    @staticmethod
    def loss_func(loss, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if loss == 'MSE':
            return np.mean((y_pred - y_true)**2)
        if loss == 'MAE':
            return np.mean(abs(y_pred - y_true))
        
    @staticmethod
    def anti_grad_loss_func(loss, y_true, y_pred):
        if loss == 'MSE':
            return 2 * (y_true - y_pred)
        if loss == 'MAE':
            return np.sign(y_true - y_pred)
    
    def fit(self, X, y, X_eval=None, y_eval=None, early_stopping=None, verbose=None):
        random.seed(self.random_state)
        self.trees = []
        self.fi = {col: 0.0 for col in X.columns}
        if self.loss == 'MSE':
            self.pred_0 = np.mean(y)
        elif self.loss == 'MAE':
            self.pred_0 = np.median(y)
        else:
            raise ValueError('Неправильный loss')
            
        pred_m = np.array([self.pred_0 for i in range(len(X))])
        count_leafs = 0 
        
        cond = (X_eval is not None and y_eval is not None and early_stopping is not None)

        best_eval_score = None
        cnt_bad = 0

        if cond:
            eval_pred = np.full(len(X_eval), self.pred_0, dtype=float)
        
        for i in range(self.n_estimators):
            count_rows = round(self.max_samples * len(X))
            count_cols = round(self.max_features * len(X.columns))
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

            diff = sample_y - sample_pred
            antigrad = MyBoostReg.anti_grad_loss_func(self.loss, sample_y, sample_pred)
            
            model = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            model.fit(sample_x, antigrad, len(X))
            model._change_leafs(model.tree, sample_x, diff, self.loss, count_leafs, self.reg)
            self.trees.append(model)
            count_leafs += model.leafs_cnt
            
            for col, val in model.fi.items():
                self.fi[col] += val

            
            if isinstance(self.learning_rate, FunctionType):
                cur_lr = self.learning_rate(i + 1)
            else:
                cur_lr = self.learning_rate

            pred_m += cur_lr * np.asarray(model.predict(X))
            
            if cond:
                eval_pred_x = np.array(model.predict(X_eval))
                eval_pred += cur_lr * eval_pred_x

            loss = MyBoostReg.loss_func(self.loss, y, pred_m)

            if cond:
                if self.metric is not None:
                    eval_score = MyBoostReg.metrics(self.metric, eval_pred, y_eval, len(y_eval))
                else:
                    eval_score = MyBoostReg.loss_func(self.loss, y_eval, eval_pred)
                
                if best_eval_score is None:
                    best_eval_score = eval_score
                    cnt_bad = 0
                else:
                    if self.metric == 'R2':
                        improved = eval_score > best_eval_score
                    else:
                        improved = eval_score < best_eval_score

                    if improved:
                        best_eval_score = eval_score
                        cnt_bad = 0
                    else:
                        cnt_bad += 1

                self.best_score = best_eval_score
                
            if self.metric is not None:
                metric_value = MyBoostReg.metrics(self.metric, pred_m, y, len(y)) if not cond else eval_score
                    
                if verbose is not None:
                    if i % verbose == 0:
                        print(f'{i}. Loss[{self.loss}]: {loss} | {self.metric}: {metric_value}')
            else:
                self.best_score = loss if not cond else eval_score

                if verbose is not None:
                    if i % verbose == 0:
                        print(f'{i}. Loss[{self.loss}]: {loss}')
                        
            if cond and cnt_bad >= early_stopping:
                del self.trees[-early_stopping:]

                self.fi = {col: 0.0 for col in X.columns}
                for tree in self.trees:
                    for col, value in tree.fi.items():
                        self.fi[col] += value

                self.best_score = best_eval_score
                break
                    
                
    
    def predict(self, X):
        if len(self.trees) == 0:
            raise ValueError('Градиентный бустинг не обучен')
        global_pred = self.pred_0
        for i, model in enumerate(self.trees):
            pred = np.array(model.predict(X))
            if isinstance(self.learning_rate, FunctionType):
                global_pred += self.learning_rate(i+1) * pred
            else:
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
        
        
    def _change_leafs(self, node, x,  diff, loss, count_leafs, reg):
        if isinstance(node, Node):
            mask_left = x[node.feature] <= node.threshold
            mask_right = x[node.feature] > node.threshold
            left = x.loc[mask_left]
            right = x.loc[mask_right]
            diff_left = diff.loc[mask_left]
            diff_right = diff.loc[mask_right]
            self._change_leafs(node.left, left, diff_left, loss, count_leafs, reg)
            self._change_leafs(node.right, right, diff_right, loss, count_leafs, reg)
        if isinstance(node, Leaf):
            if loss == 'MSE':
                node.value = np.mean(diff) + count_leafs * reg
            if loss == 'MAE':
                node.value = np.median(diff) + count_leafs * reg
                
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