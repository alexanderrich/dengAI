import pandas as pd
import numpy as np
from statsmodels.tools import eval_measures

class regularized_absloss_poisson:
    def __init__(self, train_data, X_cols, y_col, alpha=.01, lambda_reg=.1, num_iter=1000, epsilon=.001):
        self.X_cols = X_cols
        self.y_col = y_col
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        X = train_data[X_cols].as_matrix()
        y = train_data[y_col].as_matrix()
        self.num_features = X.shape[1]
        self.theta_hist = np.zeros((num_iter+1, self.num_features))
        self.loss_hist = np.zeros(num_iter+1)
        self.theta = np.zeros(self.num_features)
        self.theta_hist[0] = self.theta
        self.loss_hist[0] = self.loss(X, y, self.theta)
        for i in range(num_iter):
            self.theta = self.theta - alpha * self.gradient(X, y)
            self.theta_hist[i+1] = self.theta
            self.loss_hist[i+1] = self.loss(X, y, self.theta)


    def loss(self, X, y, theta):
        return 1.0/(len(y)) * np.sum(np.abs(np.exp(np.dot(X, theta)) - y)) + self.lambda_reg * np.dot(theta, theta)

    def gradient(self, X, y):
        approx_grad = np.zeros(self.num_features)
        for i in range(self.num_features):
            delta = np.zeros((self.num_features))
            delta[i] = self.epsilon
            hi = self.loss(X, y, self.theta + delta)
            lo = self.loss(X, y, self.theta - delta)
            approx_grad[i] = (hi - lo)/(2 * self.epsilon)
        return approx_grad

    def get_abs_error(self, test_data):
        X = test_data[self.X_cols].as_matrix()
        y = test_data[self.y_col].as_matrix()
        results = np.exp(np.dot(X, self.theta))
        test_loss = eval_measures.meanabs(results, y)
        return test_loss

    def predict(self, test_data):
        X = test_data[self.X_cols].as_matrix()
        return np.exp(np.dot(X, self.theta)).round().astype(int)
