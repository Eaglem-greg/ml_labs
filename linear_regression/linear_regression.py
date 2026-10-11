import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n=1000):
        self.lr = lr
        self.n = n
        self.bias = None
        self.weights = None
        self.loss_history = []

    def fit(self, x, y):
        samples, features = x.shape
        self.weights = np.ones(features)
        self.bias = 0.0

        for _ in range(self.n):
            y_pred = x @ self.weights + self.bias
            error = y_pred - y
            grad_w = (2 / samples) * (x.T @ error)
            grad_b = (2 / samples) * np.sum(error) 

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            self.loss_history.append(np.mean(error ** 2))
            
    def predict(self, x):
        return x @ self.weights + self.bias

class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true, dtype=float).ravel()
        self.y_pred = np.asarray(y_pred, dtype=float).ravel()
        self.data_report = {}

    def mae(self):
        self.data_report['MAE'] = self.data_report.get('MAE', 0) + float(np.mean(np.abs(self.y_pred - self.y_true)))
        return self.data_report['MAE']

    def mse(self):
        self.data_report['MSE'] = self.data_report.get('MSE', 0) + float(np.mean((self.y_pred - self.y_true)**2))
        return self.data_report['MSE']

    def r2(self):
        ss_res = np.sum((self.y_pred - self.y_true)**2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true))**2)
        if ss_tot == 0:
            return 1 if ss_res == 0 else 0
        self.data_report['R2'] = self.data_report.get('R2', 0) + 1 - ss_res / ss_tot
        return self.data_report['R2']

    def report(self):
        return f'{self.data_report}'

    def __str__(self):
        return "MAE: {MAE}, MSE: {MSE}, R2: {R2}".format(**self.data_report)

    def __repr__(self):
        return "MAE: {MAE:.4f}, MSE: {MSE:.4f}, R2: {R2:.4f}".format(**self.data_report)
    

