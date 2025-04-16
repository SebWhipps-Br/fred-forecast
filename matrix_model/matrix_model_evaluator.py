# File: matrix_model_evaluator.py (Updated)

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from factor_analyzer import Rotator
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from ea_forecast import MatrixFactorModel
from matrix_model.ea_forecast import CHOSEN_VARIABLE


class MatrixModelEvaluator:
    def __init__(self, model, country_names=None, variable_names=None, dates=None):
        """
        Initialize the MatrixModelEvaluator with a MatrixFactorModel instance.

        Parameters:
        - model: MatrixFactorModel instance
        - country_names: List of country names (default: ['Country 0', ...])
        - variable_names: List of variable names (default: ['Variable 0', ...])
        - dates: Array of datetime objects for plotting (optional)
        """
        self.model = model
        self.country_names = country_names if country_names is not None else [f"Country {i}" for i in range(model.p1)]
        self.variable_names = variable_names if variable_names is not None else [f"Variable {i}" for i in range(model.p2)]
        self.dates = dates if dates is not None else np.arange(model.T)
        self.T_train = model.T  # Training period length

    def evaluate_performance(self, true_values, predicted_values, model_name="", baseline_values=None):
        """
        Evaluate out-of-sample forecasting performance of the matrix factor model.
        """
        print(f"{model_name} Forecasting Performance:")
        print("true_values.shape:", true_values.shape)
        print("predicted_values.shape:", predicted_values.shape)

        true_vec = true_values.reshape(-1)
        pred_vec = predicted_values.reshape(-1)

        mse = mean_squared_error(true_vec, pred_vec)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_vec, pred_vec)
        mape = mean_absolute_percentage_error(true_vec, pred_vec + 1e-10) * 100  # Avoid division by zero

        print("MSE:      ", mse)
        print("RMSE:     ", rmse)
        print("MAE:      ", mae)
        print("MAPE (%): ", mape)

        n = true_vec.size
        p = self.model.k1 * self.model.k2
        ss_tot = np.sum((true_vec - np.mean(true_vec)) ** 2)
        ss_res = np.sum((true_vec - pred_vec) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
        print("Adjusted R²:", adj_r2)

        true_diff = np.sign(true_values[1:] - true_values[:-1]).reshape(-1)
        pred_diff = np.sign(predicted_values[1:] - predicted_values[:-1]).reshape(-1)
        directional_accuracy = np.mean(true_diff == pred_diff) * 100
        print("Directional Accuracy (%):", directional_accuracy)

        if baseline_values is not None and baseline_values.shape == true_values.shape:
            base_vec = baseline_values.reshape(-1)
            loss_forecast = (true_vec - pred_vec) ** 2
            loss_baseline = (true_vec - base_vec) ** 2
            d = loss_forecast - loss_baseline
            n_lags = min(2, len(d) - 1)
            hac_cov = self._newey_west_cov(d, n_lags)
            dm_stat = np.mean(d) / np.sqrt(hac_cov / len(d)) if hac_cov > 0 else np.nan
            dm_pvalue = 2 * (1 - norm.cdf(abs(dm_stat))) if not np.isnan(dm_stat) else np.nan
            print("Diebold-Mariano Test (Newey-West HAC):")
            print("  DM Statistic:", dm_stat)
            print("  P-value:", dm_pvalue)
            print("  (Positive DM stat favors baseline; negative favors forecast)")
        elif baseline_values is not None:
            print("Warning: Baseline shape mismatch, skipping DM test.")
        print()

    def evaluate_fit(self, true_values, fitted_values, model_name="", chosen_country=None, chosen_variable=None):
        """
        Evaluate in-sample fit of the matrix factor model.
        """
        print(f"{model_name} Fit Evaluation:")
        print("true_values.shape:", true_values.shape)
        print("fitted_values.shape:", fitted_values.shape)

        if true_values.shape != fitted_values.shape:
            raise ValueError("Shape mismatch between true_values and fitted_values.")

        if chosen_country is not None and chosen_variable is not None:
            country_name = self.country_names[chosen_country]
            variable_name = self.variable_names[chosen_variable]
            true_data = true_values[:, chosen_country, chosen_variable]
            fit_data = fitted_values[:, chosen_country, chosen_variable]

            print(f"Country: {country_name}, Variable: {variable_name}")
            mse = mean_squared_error(true_data, fit_data)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_data, fit_data)
            mape = mean_absolute_percentage_error(true_data, fit_data + 1e-10) * 100

            print("MSE:      ", mse)
            print("RMSE:     ", rmse)
            print("MAE:      ", mae)
            print("MAPE (%): ", mape)

            n, p = true_data.size, self.model.k1 * self.model.k2
            ss_tot = np.sum((true_data - np.mean(true_data)) ** 2)
            ss_res = np.sum((true_data - fit_data) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
            print("Adjusted R²:", adj_r2)
        else:
            true_vec = true_values.reshape(-1)
            fit_vec = fitted_values.reshape(-1)

            mse = mean_squared_error(true_vec, fit_vec)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_vec, fit_vec)
            mape = mean_absolute_percentage_error(true_vec, fit_vec + 1e-10) * 100

            print("MSE:      ", mse)
            print("RMSE:     ", rmse)
            print("MAE:      ", mae)
            print("MAPE (%): ", mape)

            n, p = true_vec.size, self.model.k1 * self.model.k2
            ss_tot = np.sum((true_vec - np.mean(true_vec)) ** 2)
            ss_res = np.sum((true_vec - fit_vec) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
            print("Adjusted R²:", adj_r2)
        print()

    def evaluate_variable_performance(self, true_values, predicted_values, chosen_country, chosen_variable,
                                     model_name="", baseline_values=None):
        """
        Evaluate forecasting performance for a specific country and variable.
        """
        country_name = self.country_names[chosen_country]
        variable_name = self.variable_names[chosen_variable]
        true_data = true_values[:, chosen_country, chosen_variable]
        pred_data = predicted_values[:, chosen_country, chosen_variable]

        print(f"{model_name} Performance for {country_name}, {variable_name}:")
        print("true_values.shape:", true_data.shape)
        print("predicted_values.shape:", pred_data.shape)

        mse = mean_squared_error(true_data, pred_data)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_data, pred_data)
        mape = mean_absolute_percentage_error(true_data, pred_data + 1e-10) * 100

        print("MSE:      ", mse)
        print("RMSE:     ", rmse)
        print("MAE:      ", mae)
        print("MAPE (%): ", mape)

        n, p = true_data.size, self.model.k1 * self.model.k2
        ss_tot = np.sum((true_data - np.mean(true_data)) ** 2)
        ss_res = np.sum((true_data - pred_data) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
        print("Adjusted R²:", adj_r2)

        true_diff = np.sign(true_data[1:] - true_data[:-1])
        pred_diff = np.sign(pred_data[1:] - pred_data[:-1])
        directional_accuracy = np.mean(true_diff == pred_diff) * 100
        print("Directional Accuracy (%):", directional_accuracy)

        if baseline_values is not None:
            base_data = baseline_values[:, chosen_country, chosen_variable]
            if base_data.shape == true_data.shape:
                loss_forecast = (true_data - pred_data) ** 2
                loss_baseline = (true_data - base_data) ** 2
                d = loss_forecast - loss_baseline
                n_lags = min(2, len(d) - 1)
                hac_cov = self._newey_west_cov(d, n_lags)
                dm_stat = np.mean(d) / np.sqrt(hac_cov / len(d)) if hac_cov > 0 else np.nan
                dm_pvalue = 2 * (1 - norm.cdf(abs(dm_stat))) if not np.isnan(dm_stat) else np.nan
                print("Diebold-Mariano Test (Newey-West HAC):")
                print("  DM Statistic:", dm_stat)
                print("  P-value:", dm_pvalue)
                print("  (Positive DM stat favors baseline; negative favors forecast)")
            else:
                print("Warning: Baseline shape mismatch, skipping DM test.")
        print()

    def _newey_west_cov(self, x, n_lags):
        """
        Compute Newey-West HAC covariance for a time series.
        """
        x = x - np.mean(x)
        n = len(x)
        cov = np.sum(x ** 2) / n
        for lag in range(1, n_lags + 1):
            weight = 1 - lag / (n_lags + 1)
            autocov = np.sum(x[lag:] * x[:-lag]) / n
            cov += 2 * weight * autocov
        return max(cov, 1e-10)

    def plot_actual_vs_common(self, actual, common_components, forecast, chosen_country, chosen_variable,
                              save_path_prefix='figures/matrix_forecast_plot'):
        """
        Plot actual data, common components, and forecasts for a specific country and variable.
        """
        country_name = self.country_names[chosen_country]
        variable_name = self.variable_names[chosen_variable]
        actual_data = actual[:, chosen_country, chosen_variable]
        common_data = common_components[:, chosen_country, chosen_variable]
        forecast_data = forecast[:, chosen_country, chosen_variable]
        h = forecast.shape[0]
        dates = self.dates

        if len(dates) != actual_data.shape[0]:
            raise ValueError(f"Dates length ({len(dates)}) must match actual data length ({actual_data.shape[0]})")
        if common_data.shape[0] != self.T_train:
            raise ValueError(f"Common components length ({common_data.shape[0]}) must match training period ({self.T_train})")
        if forecast_data.shape[0] != h:
            raise ValueError(f"Forecast length ({forecast_data.shape[0]}) must match horizon ({h})")

        plt.figure(figsize=(12, 8), dpi=300)
        plt.plot(dates, actual_data, label='Actual', color='blue', linewidth=2)
        plt.plot(dates[:self.T_train], common_data, label='Common Component', color='green', linewidth=2)
        plt.plot(dates[-h:], forecast_data, label='Forecast', color='red', linewidth=2)
        plt.axvline(x=dates[self.T_train], color='gray', linestyle='--', label='Train/Test Split')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel(f"{variable_name}", fontsize=14)
        plt.title(f'Matrix Factor Model: {country_name}, {variable_name}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        save_path = f'{save_path_prefix}_{country_name}_{variable_name}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_factor_loadings_heatmap(self, matrix='both', rotation='varimax'):
        """
        Plot heatmap of factor loadings for R (countries) and/or C (variables).
        """
        def plot_heatmap(loadings, labels, title, filename):
            plt.figure(figsize=(10, max(8, len(labels) * 0.3)))
            sns.heatmap(loadings, cmap='coolwarm', center=0,
                        xticklabels=[f'Factor {i + 1}' for i in range(loadings.shape[1])],
                        yticklabels=labels, annot=False, fmt='.2f')
            plt.title(title, fontsize=16)
            plt.xlabel('Factors', fontsize=12)
            plt.ylabel('Entities', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'figures/{filename}.png', bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()

        if matrix in ['R', 'both']:
            R_loadings = self.model.R
            if rotation == 'varimax' and R_loadings.shape[1] > 1:
                rotator = Rotator(method='varimax')
                R_loadings = rotator.fit_transform(R_loadings)
            plot_heatmap(R_loadings, self.country_names, 'Country Factor Loadings (R)', 'country_loadings')

        if matrix in ['C', 'both']:
            C_loadings = self.model.C
            if rotation == 'varimax' and C_loadings.shape[1] > 1:
                rotator = Rotator(method='varimax')
                C_loadings = rotator.fit_transform(C_loadings)
            plot_heatmap(C_loadings, self.variable_names, 'Variable Factor Loadings (C)', 'variable_loadings')

    def plot_variance_explained_per_entity(self, entity='both'):
        """
        Plot variance explained by factors for countries and/or variables.
        """
        def compute_variance_explained(loadings, factors, axis, labels, title):
            factor_variances = np.var(factors, axis=0).flatten()
            loadings_squared = loadings ** 2
            contributions = loadings_squared @ factor_variances[:loadings.shape[1]]
            total_variance = np.var(self.model.X.reshape(self.model.T, -1), axis=0).sum()
            variance_explained_pct = (contributions / total_variance) * 100

            plt.figure(figsize=(12, 6))
            plt.bar(labels, variance_explained_pct, color='skyblue', edgecolor='black')
            plt.xlabel('Entities', fontsize=12)
            plt.ylabel('Variance Explained (%)', fontsize=12)
            plt.title(title, fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.5, axis='y')
            plt.tight_layout()
            plt.savefig(f'figures/variance_explained_{entity}.png', bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()

        if entity in ['countries', 'both']:
            compute_variance_explained(self.model.R, self.model.F, axis=1, labels=self.country_names,
                                      title='Variance Explained per Country')

        if entity in ['variables', 'both']:
            compute_variance_explained(self.model.C, self.model.F, axis=2, labels=self.variable_names,
                                      title='Variance Explained per Variable')

    def plot_variance_explained_per_variable_combined(self, specific_country=None, rotation='varimax'):
        """
        Calculate and plot the variance explained for each variable, considering both country and variable loadings.
        Optionally focus on a specific country.

        Parameters:
        - specific_country: int, index of a specific country (default: None, aggregates across all countries)
        - rotation: str, 'varimax' or None for factor loadings rotation
        """
        # Get loadings and factors
        R_loadings = self.model.R  # Shape: (p1, k1)
        C_loadings = self.model.C  # Shape: (p2, k2)
        F = self.model.F           # Shape: (T, k1, k2)

        # Apply Varimax rotation if specified
        if rotation == 'varimax':
            if R_loadings.shape[1] > 1:
                rotator = Rotator(method='varimax')
                R_loadings = rotator.fit_transform(R_loadings)
            if C_loadings.shape[1] > 1:
                rotator = Rotator(method='varimax')
                C_loadings = rotator.fit_transform(C_loadings)

        # Reconstruct common component: X_hat[t, i, j] = R[i, :] @ F[t, :, :] @ C[j, :].T
        X_hat = self.model.reconstruct_X()  # Shape: (T, p1, p2)

        # Compute variance explained
        if specific_country is not None:
            # Focus on one country
            country_name = self.country_names[specific_country]
            X_var = self.model.X[:, specific_country, :]  # Shape: (T, p2)
            X_hat_var = X_hat[:, specific_country, :]     # Shape: (T, p2)
            var_total = np.var(X_var, axis=0)            # Shape: (p2,)
            var_explained = np.var(X_hat_var, axis=0)    # Shape: (p2,)
            variance_explained_pct = (var_explained / var_total * 100)  # Shape: (p2,)
            labels = self.variable_names
            title = f'Variance Explained per Variable for {country_name}'
            filename = f'variance_explained_variable_{country_name}'
        else:
            # Aggregate across all countries
            X_var = self.model.X.reshape(self.model.T, -1)  # Shape: (T, p1*p2)
            X_hat_var = X_hat.reshape(self.model.T, -1)     # Shape: (T, p1*p2)
            var_total = np.var(X_var, axis=0)              # Shape: (p1*p2,)
            var_explained = np.var(X_hat_var, axis=0)      # Shape: (p1*p2,)
            # Average variance explained per variable across countries
            var_total_per_var = var_total.reshape(self.model.p1, self.model.p2).mean(axis=0)  # Shape: (p2,)
            var_explained_per_var = var_explained.reshape(self.model.p1, self.model.p2).mean(axis=0)  # Shape: (p2,)
            variance_explained_pct = (var_explained_per_var / var_total_per_var * 100)  # Shape: (p2,)
            labels = self.variable_names
            title = 'Variance Explained per Variable (Averaged Across Countries)'
            filename = 'variance_explained_variable_combined'

        # Handle division by zero or invalid variances
        variance_explained_pct = np.nan_to_num(variance_explained_pct, nan=0.0, posinf=0.0, neginf=0.0)
        variance_explained_pct = np.clip(variance_explained_pct, 0, 100)

        # Print results
        print("Variance Explained per Variable (%):")
        for var_name, var_pct in zip(labels, variance_explained_pct):
            print(f"{var_name}: {var_pct:.2f}%")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(labels, variance_explained_pct, color='skyblue', edgecolor='black')
        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Variance Explained (%)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        plt.tight_layout()
        plt.savefig(f'figures/{filename}.png', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    def plot_factors_time_series(self, save_path='figures/matrix_factors_time_series.png'):
        """
        Plot time series of common factors (F_t) for the training period.
        """
        factors = self.model.F.reshape(self.model.T, -1)
        train_dates = self.dates[:self.T_train]

        if len(train_dates) != factors.shape[0]:
            raise ValueError(f"Dates length ({len(train_dates)}) must match factors length ({factors.shape[0]})")

        plt.figure(figsize=(12, 8), dpi=300)
        colors = plt.cm.tab10(np.linspace(0, 1, self.model.k1 * self.model.k2))
        for i in range(self.model.k1 * self.model.k2):
            plt.plot(train_dates, factors[:, i], label=f'Factor {i + 1}', color=colors[i], linewidth=1.5)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Factor Value', fontsize=12)
        plt.title('Time Series of Matrix Factors (Training Period)', fontsize=16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()


# Example usage
if __name__ == "__main__":
    # Load data
    data = np.load('processed_data/processed_3d_data.npz')
    X_unscaled = data['data']
    countries = data['countries']
    variables = data['variables']
    dates_full = data.get('dates', None)

    CHOSEN_VARIABLE = 2 # 5 = IPMN
    CHOSEN_COUNTRY = 3

    HORIZON = 12
    X_train = X_unscaled[:-HORIZON]
    X_test = X_unscaled[-HORIZON:]

    # Initialize and fit model
    model = MatrixFactorModel(X_train, k1=1, k2=4, max_iterations=1)
    model.h = HORIZON
    model.fit()

    # Initialize evaluator with full dates
    evaluator = MatrixModelEvaluator(model, country_names=countries, variable_names=variables, dates=dates_full)

    # Evaluate in-sample fit
    X_hat_train = model.reconstruct_X()
    X_train_unscaled = model._unscale(X_hat_train)
    evaluator.evaluate_fit(model._unscale(model.X), X_train_unscaled, model_name="Matrix Factor Model")
    evaluator.evaluate_fit(model._unscale(model.X), X_train_unscaled, model_name="Matrix Factor Model",
                          chosen_country=CHOSEN_COUNTRY, chosen_variable=CHOSEN_VARIABLE)

    # Evaluate forecasting performance
    X_forecast = model.forecast(HORIZON)
    evaluator.evaluate_performance(X_test, X_forecast, model_name="Matrix Factor Model")
    evaluator.evaluate_variable_performance(X_test, X_forecast, chosen_country=CHOSEN_COUNTRY, chosen_variable=CHOSEN_VARIABLE,
                                           model_name="Matrix Factor Model")

    # Visualizations
    X_full = np.concatenate([model._unscale(model.X), X_test], axis=0)
    evaluator.plot_actual_vs_common(X_full, X_train_unscaled, X_forecast, chosen_country=CHOSEN_COUNTRY, chosen_variable=CHOSEN_VARIABLE)
    evaluator.plot_factor_loadings_heatmap(matrix='both')
    evaluator.plot_variance_explained_per_entity(entity='both')
    evaluator.plot_variance_explained_per_variable_combined()  # New function
    evaluator.plot_variance_explained_per_variable_combined(specific_country=CHOSEN_COUNTRY)
    evaluator.plot_factors_time_series()