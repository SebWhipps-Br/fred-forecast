import numpy as np
from statsmodels.tsa.ar_model import AutoReg


class AutoRegressionModel:
    def __init__(self, X, h):
        """
        Initialise the AutoRegressionModel for AR(1) baseline forecasting.

        Parameters:
        - X: np.array, input data of shape (T, N)
        - h: int, forecast horizon
        """
        self.X = X
        self.h = h
        self.X_train = self.X[:-self.h]  # Training data (T-h periods)
        self.X_actual = self.X[-self.h:]  # Test data (h periods)
        self.T, self.N = X.shape
        self.T_train = self.X_train.shape[0]  # Length of training data
        self.forecasts = None
        self.fitted_values = None
        self.coefficients = None

    def fit(self):
        """Fit AR(1) models to each variable in X_train and compute fitted values and forecasts."""
        x_mean = np.mean(self.X_train, axis=0)
        xt_centered = self.X_train - x_mean

        self.forecasts = np.zeros((self.h, self.N))
        self.coefficients = np.zeros(self.N)
        self.fitted_values = np.zeros((self.T_train, self.N))  # Fitted values only for training period

        for i in range(self.N):
            X_i = xt_centered[:, i]
            model = AutoReg(X_i, lags=1, trend='n')
            result = model.fit()
            self.coefficients[i] = result.params[0]
            # Predict h steps starting from the end of training data
            self.forecasts[:, i] = result.predict(start=self.T_train, end=self.T_train + self.h - 1, dynamic=True)
            # Compute fitted values for training period (t=1 to T_train-1)
            self.fitted_values[1:, i] = self.coefficients[i] * X_i[:-1]
            self.fitted_values[0, i] = X_i[0]  # First value is actual (no lag available)

        self.fitted_values += x_mean  # Adjust back to original scale
        self.forecasts += x_mean  # Adjust forecasts back to original scale