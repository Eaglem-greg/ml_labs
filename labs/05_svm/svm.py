import numpy as np


class SVM:
    def __init__(self, task, C, epsilon, lr, n_iter, tol, patience, fit_intercept, standardize):
        self.task = task
        self.C = C
        self.epsilon = epsilon
        self.lr = lr
        self.max_iter = n_iter
        self.tol = tol
        self.patience = patience
        self.fit_intercept = fit_intercept
        self.standardize = standardize

        self.coef = None
        self.intercept = None
        self.classes = None
        self.mean = None
        self.scale = None
        self.loss_history = None
        self.n_iter = 0

    def fit_scaler(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale = np.where(self.scale == 0, 1.0, self.scale)

    def transform(self, X):
        if self.standardize:
            return (X - self.mean) / self.scale
        return X

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.standardize:
            self.fit_scaler(X)
            X = self.transform(X)

        if self.task == "classification":
            self.fit_classification(X, y)
        else:
            self.fit_regression(X, y)

        return self

    def fit_classification(self, X, y):
        self.classes = np.unique(y)

        weights = []
        intercepts = []
        self.loss_history = {}
        self.n_iter = {}

        for label in self.classes:
            y_binary = np.where(y == label, 1.0, -1.0)
            w, b, history, iterations = self.fit_binary_svm(X, y_binary)

            weights.append(w)
            intercepts.append(b)
            self.loss_history[label] = history
            self.n_iter[label] = iterations

        self.coef = np.vstack(weights)
        self.intercept = np.asarray(intercepts)

    def fit_binary_svm(self, X, y):
        samples, features = X.shape
        w = np.zeros(features)
        b = 0.0

        history = []
        previous_loss = None
        stable_iterations = 0

        for iteration in range(self.max_iter):
            scores = X @ w + b
            margins = y * scores
            violations = margins < 1

            grad_w = w - (self.C / samples) * (X[violations].T @ y[violations])
            grad_b = -(self.C / samples) * np.sum(y[violations])

            w -= self.lr * grad_w
            if self.fit_intercept:
                b -= self.lr * grad_b

            loss = self.binary_loss(X, y, w, b)
            history.append(loss)

            if previous_loss is not None and abs(previous_loss - loss) < self.tol:
                stable_iterations += 1
            else:
                stable_iterations = 0

            if stable_iterations >= self.patience:
                return w, b, history, iteration + 1

            previous_loss = loss

        return w, b, history, self.max_iter

    def binary_loss(self, X, y, w, b):
        margins = y * (X @ w + b)
        hinge_loss = np.maximum(0.0, 1.0 - margins)
        return 0.5 * np.dot(w, w) + self.C * np.mean(hinge_loss)

    def fit_regression(self, X, y):
        samples, features = X.shape
        w = np.zeros(features)
        b = 0.0

        self.loss_history = []
        previous_loss = None
        stable_iterations = 0

        for iteration in range(self.max_iter):
            predictions = X @ w + b
            errors = y - predictions
            outside_tube = np.abs(errors) > self.epsilon

            error_signs = np.sign(errors[outside_tube])
            grad_w = w - (self.C / samples) * (X[outside_tube].T @ error_signs)
            grad_b = -(self.C / samples) * np.sum(error_signs)

            w -= self.lr * grad_w
            if self.fit_intercept:
                b -= self.lr * grad_b

            loss = self.regression_loss(X, y, w, b)
            self.loss_history.append(loss)

            if previous_loss is not None and abs(previous_loss - loss) < self.tol:
                stable_iterations += 1
            else:
                stable_iterations = 0

            if stable_iterations >= self.patience:
                self.n_iter = iteration + 1
                break

            previous_loss = loss
        else:
            self.n_iter = self.max_iter

        self.coef = w
        self.intercept = b

    def regression_loss(self, X, y, w, b):
        errors = y - (X @ w + b)
        epsilon_loss = np.maximum(0.0, np.abs(errors) - self.epsilon)
        return 0.5 * np.dot(w, w) + self.C * np.mean(epsilon_loss)

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        X = self.transform(X)

        if self.task == "classification":
            return X @ self.coef.T + self.intercept
        return X @ self.coef + self.intercept

    def predict(self, X):
        values = self.decision_function(X)

        if self.task == "classification":
            class_indices = np.argmax(values, axis=1)
            return self.classes[class_indices]
        return values

