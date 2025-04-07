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
        self.T, self.N = X.shape
        self.forecasts = None
        self.fitted_values = None
        self.coefficients = None

    def fit(self):
        """Fit AR(1) models to each variable and compute fitted values and forecasts."""
        x_mean = np.mean(self.X, axis=0)
        xt_centered = self.X - x_mean

        self.forecasts = np.zeros((self.h, self.N))
        self.coefficients = np.zeros(self.N)
        self.fitted_values = np.zeros((self.T, self.N))

        for i in range(self.N):
            X_i = xt_centered[:, i]
            model = AutoReg(X_i, lags=1, trend='n')
            result = model.fit()
            self.coefficients[i] = result.params[0]
            self.forecasts[:, i] = result.predict(start=self.T, end=self.T + self.h - 1, dynamic=True)
            self.fitted_values[1:, i] = self.coefficients[i] * X_i[:-1]
            self.fitted_values[0, i] = X_i[0]

        self.fitted_values += x_mean
        self.forecasts += x_mean  # Adjust forecasts back to original scale