import numpy as np

class PolynomialRegression:
    def __init__(self, standart, intercept, lr=0.01, n_iter=1000, degree=1):
        self.lr =lr
        self.n_iter = n_iter
        self.degree = degree
        self.fit_standart = standart
        self.fit_intercept = intercept
        self.combination_list = []
        self.loss_history = []
        self.weights = None
        self.bias = None
        self.mean = None
        self.scale = None

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        m, n = X.shape
        if m < 1 or n < 1:
            raise ValueError("X не должна быть пустой")
        
        n_features = X.shape[1]
        combination_list = []

        def generate_combinations(start, length, current):
            if length == 0:
                combination_list.append(tuple(current))
                return

            for feature_index in range(start, n_features):
                current.append(feature_index)

                generate_combinations(
                    start=feature_index,
                    length=length - 1,
                    current=current
                )

                current.pop()

        for current_degree in range(1, self.degree + 1):
            generate_combinations(
                start=0,
                length=current_degree,
                current=[]
            )

        polynomial_columns = []

        for combination in combination_list:
            column = np.ones(X.shape[0])

            for feature_index in combination:
                column *= X[:, feature_index]

            polynomial_columns.append(column)

        self.combination_list = combination_list

        return np.column_stack(polynomial_columns)


    def fit(self, x, y):
        
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x_poly = self.transform(x)

        if self.fit_standart:
            self.mean = np.mean(x_poly, axis=0)
            self.scale = np.std(x_poly, axis=0)

            self.scale[self.scale == 0] = 1

            x_poly = (x_poly - self.mean) / self.scale


        samples, features = x_poly.shape

        self.weights = np.zeros(features)
        self.bias = 0.0
        self.loss_history.clear()

        for _ in range(self.n_iter):
            y_pred = x_poly @ self.weights

            if self.fit_intercept:
                y_pred = y_pred + self.bias

            error = y_pred - y

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            grad_w = ((2 / samples) * (x_poly.T @ error))

            self.weights -= self.lr * grad_w

            if self.fit_intercept:
                grad_b = ((2 / samples) * np.sum(error))

                self.bias -= self.lr * grad_b

                
    def predict(self, x):
        x_transform = self.transform(x)
        if self.fit_standart:
            x_transform = (x_transform - self.mean) / self.scale
        return x_transform @ self.weights + (self.bias if self.fit_intercept else 0.0)


class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true, dtype=float).ravel()
        self.y_pred = np.asarray(y_pred, dtype=float).ravel()
        self.data_report = {}

    def mae(self):
        value = float(np.mean(np.abs(self.y_pred - self.y_true)))
        self.data_report["MAE"] = value
        return value

    def mse(self):
        value = float(np.mean((self.y_pred - self.y_true) ** 2))
        self.data_report["MSE"] = value
        return value

    def r2(self):
        ss_res = np.sum((self.y_pred - self.y_true) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)

        if ss_tot == 0:
            value = 1.0 if ss_res == 0 else 0.0
        else:
            value = float(1 - ss_res / ss_tot)

        self.data_report["R2"] = value
        return value

    def report(self):
        return self.data_report

    def __str__(self):
        return " ".join(f"{name}: {value}" for name, value in self.data_report.items())

    def __repr__(self):
        return self.__str__()