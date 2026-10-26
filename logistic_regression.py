import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n=1000):
        self.lr = lr
        self.n = n
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, Y):
        X = np.asarray(X, float)
        samples, features = X.shape
        self.weights = np.ones(features)
        self.bias = 0.0

        for _ in range(self.n):
            z = X @ self.weights + self.bias
            p = 1 / (1 + np.exp(-z))
            grad_w = (1 / samples) * (X.T @ (p - Y))
            grad_b = (1 / samples) * np.sum(p - Y)
            self.weights -= self.lr*grad_w
            self.bias -= self.lr*grad_b
            self.loss_history.append((-1/samples) * np.sum(Y*np.log(p)+ (1-Y)*np.log(1-p)))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.weights + self.bias
        p = 1 / (1 + np.exp(-z))
        classes = (p >= 0.5).astype(int)
        return p, classes


class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true).reshape(-1)
        self.y_pred = np.asarray(y_pred).reshape(-1)
        self.tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.fn = np.sum((self.y_true == 1) & (self.y_pred == 0))


    def accuracy(self):
        correct = np.sum(self.y_true == self.y_pred)
        total = len(self.y_true)

        return correct / total

    def recall(self):
        denominator =  self.tp + self.fn
        if denominator == 0:
            return 0.0

        return self.tp / denominator

    def precision(self):
        denominator = self.tp + self.fp
        if denominator == 0:
            return 0.0

        return self.tp / denominator

    def f_score(self):
        precision = self.precision()
        recall = self.recall()
        denominator = precision + recall
        if denominator == 0:
            return 0.0

        return 2 * precision * recall / denominator

    def confusion_matrix(self):
        return np.array([
            [self.tn, self.fp],
            [self.fn, self.tp]
        ])
