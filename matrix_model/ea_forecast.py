# File: matrix_factor_model.py (Updated)

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class MatrixFactorModel:
    def __init__(self, X, k1=None, k2=None, max_k1=None, max_k2=None, max_iterations=1, forecast_method='linear'):
        """
        Initialise the matrix factor model with standardized data.

        Parameters:
        - X: 3D array of shape (T, p1, p2) - time x rows x columns
        - k1: Number of row factors (for R)
        - k2: Number of column factors (for C)
        - max_iterations: Number of projection iterations (default 1)
        - forecast_method: str, 'linear' for Linear Regression or 'rf' for Random Forest
        """
        if forecast_method not in ['linear', 'rf']:
            raise ValueError("forecast_method must be 'linear' or 'rf'")
        self.forecast_method = forecast_method
        self.X_unscaled = X
        self.X = self._standardise_data(X)
        self.T, self.p1, self.p2 = X.shape
        self.max_iterations = max_iterations

        self.R = None
        self.C = None
        self.F = None

        if k1 is None or k2 is None:
            if max_k1 is None or max_k2 is None:
                raise ValueError("Must provide max_k1 and max_k2 if k1 or k2 is None")
            print("Determining optimal number of factors using ER method...")
            self.k1, self.k2 = self.determine_factors(max_k1, max_k2)
        else:
            self.k1, self.k2 = k1, k2

    def _standardise_data(self, X):
        """
        Standardises the input data along the time axis (mean 0, std 1).
        """
        X_scaled = X.copy().astype(float)
        mean = np.mean(X_scaled, axis=0, keepdims=True)
        std = np.std(X_scaled, axis=0, keepdims=True)
        X_scaled = (X_scaled - mean) / std
        return X_scaled

    def _unscale(self, X_scaled):
        """
        Unscale standardized data to original units.
        """
        mean = np.mean(self.X_unscaled, axis=0, keepdims=True)
        std = np.std(self.X_unscaled, axis=0, keepdims=True)
        return X_scaled * std + mean

    def _compute_M(self, axis='rows'):
        """
        Computes the scaled sample covariance matrix M̂.
        """
        if axis == 'rows':
            M = np.zeros((self.p1, self.p1))
            for t in range(self.T):
                Xt = self.X[t]
                M += Xt @ Xt.T
            M /= (self.T * self.p1 * self.p2)
        elif axis == 'columns':
            M = np.zeros((self.p2, self.p2))
            for t in range(self.T):
                Xt = self.X[t]
                M += Xt.T @ Xt
            M /= (self.T * self.p1 * self.p2)
        else:
            raise NotImplementedError
        return M

    def _compute_pca_loading(self, M, n_components, scale_factor):
        """
        Helper to compute PCA-based loading matrix.
        """
        pca = PCA(n_components=n_components)
        pca.fit(M)
        loading = np.sqrt(scale_factor) * pca.components_.T
        print(f"Shape: {loading.shape}, Explained variance: {pca.explained_variance_ratio_}")
        return loading

    def initial_estimate(self):
        """
        Compute initial estimates R̂ and Ĉ using PCA.
        """
        self.R = self._compute_pca_loading(self._compute_M('rows'), self.k1, self.p1)
        self.C = self._compute_pca_loading(self._compute_M('columns'), self.k2, self.p2)

    def _project_data(self):
        """
        Project X into lower-dimensional Ŷ_t and Ẑ_t.
        """
        Y_hat = np.zeros((self.T, self.p1, self.k2))
        Z_hat = np.zeros((self.T, self.p2, self.k1))
        for t in range(self.T):
            Xt = self.X[t]
            Y_hat[t] = (Xt @ self.C) / self.p2
            Z_hat[t] = (Xt.T @ self.R) / self.p1
        return Y_hat, Z_hat

    def _compute_covariance(self, data, dim):
        """
        Compute covariance matrix from projected data.
        """
        M = np.zeros((dim, dim))
        for t in range(self.T):
            M += data[t] @ data[t].T
        M /= (self.T * dim)
        return M

    def _refine_estimates(self, Y_hat, Z_hat):
        """
        Refine R and C using projected data.
        """
        self.R = self._compute_pca_loading(self._compute_covariance(Y_hat, self.p1), self.k1, self.p1)
        self.C = self._compute_pca_loading(self._compute_covariance(Z_hat, self.p2), self.k2, self.p2)

    def fit(self):
        """
        Fits the matrix factor model.
        """
        print("Computing initial estimates...")
        self.initial_estimate()
        for i in range(self.max_iterations):
            print(f"Iteration {i + 1}/{self.max_iterations}")
            Y_hat, Z_hat = self._project_data()
            self._refine_estimates(Y_hat, Z_hat)
        self.F = self.estimate_factors()

    def get_loading_matrices(self):
        """
        Return the estimated R and C.
        """
        return self.R, self.C

    def estimate_factors(self):
        """
        Estimate F_t for each time t using scaled transposes.
        """
        R_T_scaled = self.R.T / self.p1
        C_T_scaled = self.C.T / self.p2
        F = np.zeros((self.T, self.k1, self.k2))
        for t in range(self.T):
            Xt = self.X[t]
            F[t] = R_T_scaled @ Xt @ C_T_scaled.T
        return F

    def reconstruct_X(self):
        """
        Reconstruct X̂_t = R F_t C^T for each t.
        """
        F = self.estimate_factors()
        X_hat = np.zeros((self.T, self.p1, self.p2))
        C_T = self.C.T
        for t in range(self.T):
            X_hat[t] = self.R @ F[t] @ C_T
        return X_hat

    def compute_residuals(self):
        """
        Compute residuals E_t = X_t - X̂_t.
        """
        X_hat = self.reconstruct_X()
        E = self.X - X_hat
        return E

    def check_model(self):
        """
        Check model fit with diagnostics.
        """
        X_hat = self.reconstruct_X()
        E = self.compute_residuals()

        residual_norms = np.linalg.norm(E, ord='fro', axis=(1, 2))
        avg_residual_norm = np.mean(residual_norms)

        total_variance = np.sum(np.linalg.norm(self.X, ord='fro', axis=(1, 2)) ** 2)
        explained_variance = np.sum(np.linalg.norm(X_hat, ord='fro', axis=(1, 2)) ** 2)
        residual_variance = np.sum(np.linalg.norm(E, ord='fro', axis=(1, 2)) ** 2)
        explained_variance_ratio = explained_variance / total_variance

        print(f"\nModel Diagnostics:")
        print(f"Average Frobenius norm of residuals: {avg_residual_norm:.4f}")
        print(f"Explained variance ratio: {explained_variance_ratio:.4f}")
        print(f"Residual variance ratio: {residual_variance / total_variance:.4f}")

        return X_hat, E, residual_norms

    def forecast(self, h):
        """
        Forecast X_{t+h} using structured matrix factor model with the specified method.

        Parameters:
        - h: int, forecast horizon

        Returns:
        - np.ndarray, forecasted data of shape (h, p1, p2)
        """
        # Vectorize F and X
        F_vec = self.F.reshape(self.T, -1)  # Shape: (T, k1*k2)
        X_vec = self.X.reshape(self.T, -1)  # Shape: (T, p1*p2)
        X_forecast_vec = np.zeros((h, self.p1 * self.p2))  # Shape: (h, p1*p2)

        # Kronecker product of C and R
        R_kron_C = np.kron(self.C, self.R)  # Shape: (p1*p2, k1*k2)

        for horizon in range(1, h + 1):
            if self.T - horizon < 1:
                raise ValueError(f"Not enough data for horizon {horizon}.")
            F_target = F_vec[horizon:]  # Shape: (T-horizon, k1*k2)
            F_t = F_vec[:-horizon]  # Shape: (T-horizon, k1*k2)

            # Fit dynamics for F_t
            if self.forecast_method == 'linear':
                model = LinearRegression()
            else:  # self.forecast_method == 'rf'
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(F_t, F_target)

            # For LinearRegression, use coef_; for RandomForest, predict next step
            if self.forecast_method == 'linear':
                A = model.coef_  # Shape: (k1*k2, k1*k2)
            else:
                last_factors = F_vec[-horizon:, :].reshape(-1, self.k1 * self.k2)
                F_pred = model.predict(last_factors)
                A = np.eye(self.k1 * self.k2)  # Placeholder for RF

            # Compute structured B
            B = R_kron_C @ A if self.forecast_method == 'linear' else R_kron_C

            # Forecast using latest factors
            last_factors = F_vec[-1, :].reshape(1, -1)  # Shape: (1, k1*k2)
            if self.forecast_method == 'linear':
                X_forecast_vec[horizon - 1] = (B @ last_factors.T).T  # Shape: (1, p1*p2)
            else:
                F_pred_last = model.predict(last_factors)  # Shape: (1, k1*k2)
                X_forecast_vec[horizon - 1] = (B @ F_pred_last.T).T  # Shape: (1, p1*p2)

        X_forecast = X_forecast_vec.reshape(h, self.p1, self.p2)
        return self._unscale(X_forecast)

    def plot_variable_across_countries(self, variable_idx, h, X_test, dates_full, country_names=None):
        """
        Plot a variable across all countries using Plotly, showing full time series with forecast and test data.

        Parameters:
        - variable_idx: Index of the variable to plot (0 to p2-1)
        - h: Forecast horizon (used to split training/test)
        - X_test: Test data array of shape (h, p1, p2)
        - dates_full: Array of dates for the full time series (length T + h)
        - country_names: List of country names (optional, defaults to indices)
        """
        if variable_idx < 0 or variable_idx >= self.p2:
            raise ValueError(f"variable_idx must be between 0 and {self.p2 - 1}")
        if len(dates_full) != self.T + h:
            raise ValueError(f"dates_full length ({len(dates_full)}) must match T + h ({self.T + h})")

        # Full original data (training + test)
        X_orig_train = self.X_unscaled  # Shape: (T, p1, p2)
        X_orig_test = X_test  # Shape: (h, p1, p2)
        X_orig_full = np.concatenate([X_orig_train, X_orig_test], axis=0)  # Shape: (T+h, p1, p2)

        # Reconstructed training data
        X_hat_train = self.reconstruct_X()  # Shape: (T, p1, p2)
        mean = np.mean(self.X_unscaled, axis=0, keepdims=True)
        std = np.std(self.X_unscaled, axis=0, keepdims=True)
        X_hat_train_unscaled = X_hat_train * std + mean

        # Forecast data
        X_forecast = self.forecast(h)  # Shape: (h, p1, p2)

        # Extract variable across all countries
        X_full = X_orig_full[:, :, variable_idx]  # Shape: (T+h, p1)
        X_hat_train_var = X_hat_train_unscaled[:, :, variable_idx]  # Shape: (T, p1)
        X_forecast_var = X_forecast[:, :, variable_idx]  # Shape: (h, p1)

        # Dates
        dates_train = dates_full[:self.T]
        dates_test = dates_full[self.T:]

        # Country names
        if country_names is None or len(country_names) != self.p1:
            country_names = [f"Country {i}" for i in range(self.p1)]

        # Create Plotly figure
        fig = go.Figure()

        for i in range(self.p1):
            # Full actual data (solid line)
            fig.add_trace(go.Scatter(
                x=dates_full, y=X_full[:, i], mode='lines', name=f"{country_names[i]} (Actual)",
                line=dict(color='blue', width=1), opacity=0.5, legendgroup=country_names[i]
            ))
            # Reconstructed training data (dashed line)
            fig.add_trace(go.Scatter(
                x=dates_train, y=X_hat_train_var[:, i], mode='lines', name=f"{country_names[i]} (Model)",
                line=dict(color='orange', dash='dash', width=1), opacity=0.7, legendgroup=country_names[i],
                showlegend=False
            ))
            # Forecast data (dotted line)
            fig.add_trace(go.Scatter(
                x=dates_test, y=X_forecast_var[:, i], mode='lines', name=f"{country_names[i]} (Forecast)",
                line=dict(color='red', dash='dot', width=1), opacity=0.7, legendgroup=country_names[i],
                showlegend=False
            ))

        # Add vertical line for train/test split
        fig.add_vline(x=dates_full[self.T], line=dict(color='gray', dash='dash'), name='Train/Test Split')

        # Update layout
        method_name = 'Linear Regression' if self.forecast_method == 'linear' else 'Random Forest'
        fig.update_layout(
            title=f"Variable {variable_idx} Across Countries: {method_name}",
            xaxis_title="Date",
            yaxis_title="Value",
            legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
            width=1600, height=900
        )

        fig.show()

    def _compute_eigenvalue_ratios(self, M, max_factors):
        """
        Compute eigenvalue ratios for a covariance matrix.
        """
        pca = PCA(n_components=min(max_factors, M.shape[0]))
        pca.fit(M)
        eigenvalues = pca.explained_variance_
        if len(eigenvalues) < max_factors:
            eigenvalues = np.pad(eigenvalues, (0, max_factors - len(eigenvalues)), 'constant')
        ratios = eigenvalues[:-1] / eigenvalues[1:]
        return ratios

    def _er_test(self, ratios):
        """
        Determine the number of factors using the eigenvalue-ratio test.
        """
        if len(ratios) < 1:
            return 1
        max_ratio_idx = np.argmax(ratios)
        return max_ratio_idx + 1

    def determine_factors(self, max_k1, max_k2):
        """
        Determine optimal number of row (k1) and column (k2) factors using ER method.
        """
        M_rows = self._compute_M(axis='rows')
        row_ratios = self._compute_eigenvalue_ratios(M_rows, max_k1)
        k1 = self._er_test(row_ratios)
        print(f"Row eigenvalue ratios: {row_ratios[:min(5, len(row_ratios))]}...")
        print(f"Optimal k1 (row factors): {k1}")

        M_cols = self._compute_M(axis='columns')
        col_ratios = self._compute_eigenvalue_ratios(M_cols, max_k2)
        k2 = self._er_test(col_ratios)
        print(f"Column eigenvalue ratios: {col_ratios[:min(5, len(col_ratios))]}...")
        print(f"Optimal k2 (column factors): {k2}")

        return k1, k2


# Load and run
if __name__ == "__main__":
    data = np.load('processed_data/processed_3d_data.npz')
    X_unscaled = data['data']
    countries = data['countries']
    variables = data['variables']
    dates = data.get('dates', None)

    T, n_countries, n_variables = X_unscaled.shape

    MAX_COUNTRY_FACTORS = 5
    MAX_VARIABLE_FACTORS = 10
    HORIZON = 12
    CHOSEN_VARIABLE = 21

    X_train = X_unscaled[:-HORIZON]
    X_test = X_unscaled[-HORIZON:]
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # Instantiate and run Linear Regression model
    print("\nRunning Linear Regression Model...")
    model_linear = MatrixFactorModel(X_train, max_k1=MAX_COUNTRY_FACTORS, max_k2=MAX_VARIABLE_FACTORS,
                                     max_iterations=1, forecast_method='linear')
    model_linear.fit()
    R_linear, C_linear = model_linear.get_loading_matrices()
    print("\nLinear Model - Final R (countries x factors):\n", R_linear[:5, :])
    print("Linear Model - Final C (variables x factors):\n", C_linear[:5, :])
    X_hat_train_linear, E_train_linear, residual_norms_train_linear = model_linear.check_model()

    # Instantiate and run Random Forest model
    print("\nRunning Random Forest Model...")
    model_rf = MatrixFactorModel(X_train, max_k1=MAX_COUNTRY_FACTORS, max_k2=MAX_VARIABLE_FACTORS,
                                 max_iterations=1, forecast_method='rf')
    model_rf.fit()
    R_rf, C_rf = model_rf.get_loading_matrices()
    print("\nRandom Forest Model - Final R (countries x factors):\n", R_rf[:5, :])
    print("Random Forest Model - Final C (variables x factors):\n", C_rf[:5, :])
    X_hat_train_rf, E_train_rf, residual_norms_train_rf = model_rf.check_model()

    # Generate plots
    print("\nGenerating Linear Regression plot...")
    model_linear.plot_variable_across_countries(variable_idx=CHOSEN_VARIABLE, h=HORIZON, X_test=X_test,
                                                dates_full=dates, country_names=countries)
    print("\nGenerating Random Forest plot...")
    model_rf.plot_variable_across_countries(variable_idx=CHOSEN_VARIABLE, h=HORIZON, X_test=X_test,
                                            dates_full=dates, country_names=countries)