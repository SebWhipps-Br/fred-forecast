import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from factor_analyzer import Rotator
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

class ModelEvaluator:
    def __init__(self, model):
        """
        Initialise the ModelEvaluator with a StaticFactorModel instance.

        Parameters:
        - model: StaticFactorModel instance
        """
        self.model = model

    def evaluate_performance(self, true_values, predicted_values, model_name="", baseline_values=None):
        print(f"{model_name} Performance:")
        print("true_values.shape:", true_values.shape)
        print("predicted_values.shape:", predicted_values.shape)

        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predicted_values)
        mape = mean_absolute_percentage_error(true_values, predicted_values)

        print("MSE:      ", mse)
        print("RMSE:     ", rmse)
        print("MAE:      ", mae)
        print("MAPE (%): ", mape)

        n, p = true_values.shape[0], self.model.q
        ss_tot = np.sum((true_values - np.mean(true_values, axis=0)) ** 2)
        ss_res = np.sum((true_values - predicted_values) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print("R²:", r2)

        true_diff = np.sign(true_values[1:] - true_values[:-1])
        pred_diff = np.sign(predicted_values[1:] - predicted_values[:-1])
        directional_accuracy = np.mean(true_diff == pred_diff) * 100
        print("Directional Accuracy (%):", directional_accuracy)

        if baseline_values is not None and baseline_values.shape == true_values.shape:
            loss_forecast = np.mean((true_values - predicted_values) ** 2, axis=1)
            loss_baseline = np.mean((true_values - baseline_values) ** 2, axis=1)
            d = loss_forecast - loss_baseline
            # Manual Newey-West HAC covariance
            n_lags = min(2, len(d) - 1)  # Adjust lags for short horizon
            hac_cov = self._newey_west_cov(d, n_lags)
            dm_stat = np.mean(d) / np.sqrt(hac_cov / len(d))
            dm_pvalue = 2 * (1 - norm.cdf(abs(dm_stat)))
            print("Diebold-Mariano Test (Newey-West HAC):")
            print("  DM Statistic:", dm_stat)
            print("  P-value:", dm_pvalue)
            print("  (Positive DM stat favors baseline; negative favors forecast)")
        elif baseline_values is not None:
            print("Warning: Baseline shape mismatch, skipping DM test.")
        print()

    def evaluate_fit(self, true_values, fitted_values, model_name="", chosen_component=None):
        """
        Evaluate the in-sample fit of the model.

        Parameters:
        - true_values: np.array, actual training data (T_train, N) for static models or (T, N) for AR(1)
        - fitted_values: np.array, fitted values from the model (T_train, N) or (T, N)
        - model_name: str, name of the model for printing
        - chosen_component: int or None, index of the variable to evaluate (if None, evaluates multivariately)
        """
        print(f"{model_name} Fit Evaluation:")
        print("true_values.shape:", true_values.shape)
        print("fitted_values.shape:", fitted_values.shape)

        if true_values.shape != fitted_values.shape:
            raise ValueError("Shape mismatch between true_values and fitted_values.")

        if chosen_component is None:
            # Multivariate fit (all variables)
            mse = mean_squared_error(true_values, fitted_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_values, fitted_values)
            mape = mean_absolute_percentage_error(true_values, fitted_values)

            print("MSE:      ", mse)
            print("RMSE:     ", rmse)
            print("MAE:      ", mae)
            print("MAPE (%): ", mape)

            n, p = true_values.shape[0], self.model.q if hasattr(self.model, 'q') else 1  # p=1 for AR(1)
            ss_tot = np.sum((true_values - np.mean(true_values, axis=0)) ** 2)
            ss_res = np.sum((true_values - fitted_values) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            print("R²:", r2)
        else:
            # Variable-specific fit
            variable_name = self.model.df.columns[chosen_component]
            true_var = true_values[:, chosen_component]
            fit_var = fitted_values[:, chosen_component]

            print(f"Variable: {variable_name}")
            print("true_values.shape:", true_var.shape)
            print("fitted_values.shape:", fit_var.shape)

            mse = mean_squared_error(true_var, fit_var)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_var, fit_var)
            mape = mean_absolute_percentage_error(true_var, fit_var)

            print("MSE:      ", mse)
            print("RMSE:     ", rmse)
            print("MAE:      ", mae)
            print("MAPE (%): ", mape)

            n, p = true_var.shape[0], self.model.q if hasattr(self.model, 'q') else 1
            ss_tot = np.sum((true_var - np.mean(true_var)) ** 2)
            ss_res = np.sum((true_var - fit_var) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            print("R²:", r2)
        print()

    def evaluate_variable_performance(self, true_values, predicted_values, chosen_component, model_name="",
                                      baseline_values=None):
        variable_name = self.model.df.columns[chosen_component]
        true_var = true_values[:, chosen_component]
        pred_var = predicted_values[:, chosen_component]

        print(f"{model_name} Performance for {variable_name}:")
        print("true_values.shape:", true_var.shape)
        print("predicted_values.shape:", pred_var.shape)

        mse = mean_squared_error(true_var, pred_var)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_var, pred_var)
        mape = mean_absolute_percentage_error(true_var, pred_var)

        print("MSE:      ", mse)
        print("RMSE:     ", rmse)
        print("MAE:      ", mae)
        print("MAPE (%): ", mape)

        n, p = true_var.shape[0], self.model.q
        ss_tot = np.sum((true_var - np.mean(true_var)) ** 2)
        ss_res = np.sum((true_var - pred_var) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print("R²:", r2)

        true_diff = np.sign(true_var[1:] - true_var[:-1])
        pred_diff = np.sign(pred_var[1:] - pred_var[:-1])
        directional_accuracy = np.mean(true_diff == pred_diff) * 100
        print("Directional Accuracy (%):", directional_accuracy)

        if baseline_values is not None:
            baseline_var = baseline_values[:, chosen_component]
            if baseline_var.shape == true_var.shape:
                loss_forecast = (true_var - pred_var) ** 2
                loss_baseline = (true_var - baseline_var) ** 2
                d = loss_forecast - loss_baseline
                n_lags = min(2, len(d) - 1)
                hac_cov = self._newey_west_cov(d, n_lags)
                dm_stat = np.mean(d) / np.sqrt(hac_cov / len(d))
                dm_pvalue = 2 * (1 - norm.cdf(abs(dm_stat)))
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

        Parameters:
        - x: np.array, 1D array of loss differentials
        - n_lags: int, number of lags for HAC adjustment

        Returns:
        - float, HAC covariance estimate
        """
        x = x - np.mean(x)  # Center the series
        n = len(x)
        # Simple variance (lag 0)
        cov = np.sum(x ** 2) / n
        # Add autocovariances with Bartlett weights
        for lag in range(1, n_lags + 1):
            weight = 1 - lag / (n_lags + 1)
            autocov = np.sum(x[lag:] * x[:-lag]) / n
            cov += 2 * weight * autocov  # Symmetric, so multiply by 2
        return cov

    def plot_actual_vs_common(self, actual, common_components, forecast, chosen_component,
                              save_path_prefix='figures/forecast_plot'):
        """Generate a high-resolution plot comparing actual data, common components, and forecasts."""
        i = chosen_component
        variable_name = self.model.df.columns[i]
        mask = self.model.dates >= '2000-01-01'
        plot_dates = self.model.dates[mask]
        actual_data = actual[mask, i]
        common_data = common_components[mask[:-self.model.h], i]
        forecast_values = forecast[:, i]

        model_name = "Static Factor Linear Regression Forecast" if self.model.forecast_method == 'linear' else "Static Factor Random Forest Forecast"

        plt.figure(figsize=(12, 8), dpi=300)
        plt.plot(plot_dates, actual_data, label='Actual', color='blue', linewidth=2)
        plt.plot(plot_dates[:-self.model.h], common_data, label='Common Component', color='green', linewidth=2)
        plt.plot(plot_dates[-self.model.h:], forecast_values, label='Forecast', color='red', linewidth=2)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel(variable_name, fontsize=14)
        plt.title(f'{model_name} Performance for {variable_name}',
                  fontsize=16, pad=15)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        save_path = f'{save_path_prefix}_{model_name.lower().replace(" ", "_")}_{variable_name}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_actual_vs_common_plotly(self, actual, common_components, forecast, chosen_component):
        """Plot using Plotly."""
        i = chosen_component
        actual_data = actual[:, i]
        common_data = common_components[:, i]
        forecast_values = forecast[:, i]
        plot_dates = self.model.dates

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_dates, y=actual_data, mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=plot_dates[:-self.model.h], y=common_data, mode='lines', name='Common Component',
                                 line=dict(color='green')))
        fig.add_trace(go.Scatter(x=plot_dates[-self.model.h:], y=forecast_values, mode='lines', name='Forecast',
                                 line=dict(color='red', dash='dash')))
        fig.update_layout(
            title_text=f"Actual vs Common Components with Forecast for Variable {i + 1}",
            xaxis_title="Date",
            yaxis_title=f"Variable {i + 1}",
            autosize=False,
            width=1000,
            height=600,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
            paper_bgcolor="LightSteelBlue",
        )
        fig.show()

    def plot_factor_loadings_heatmap(self):
        """Heatmap of factor loadings for all variables with Varimax rotation."""
        if self.model.pca is None:
            raise ValueError("Run fit_static_model() first to fit the PCA model.")

        # Original PCA loadings
        loadings = self.model.pca.components_.T  # Shape: (n, q)

        # Apply Varimax rotation
        rotator = Rotator(method='varimax')
        rotated_loadings = rotator.fit_transform(loadings)

        # Plot heatmap
        plt.figure(figsize=(20, 30))
        sns.heatmap(rotated_loadings, cmap='coolwarm', center=0,
                    xticklabels=[f'Factor {i + 1}' for i in range(self.model.q)],
                    yticklabels=self.model.df.columns,
                    annot=False, fmt='.2f')
        plt.title('Varimax-Rotated Factor Loadings Heatmap', fontsize=16)
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('Variables', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_variable_factor_contribution(self, chosen_variable):
        """Plot factor contributions to a chosen variable's variance with Varimax rotation."""
        rotator = Rotator(method='varimax')
        rotated_loadings = rotator.fit_transform(self.model.pca.components_.T)
        factor_variances = np.var(self.model.factors, axis=0)
        contributions = (rotated_loadings[chosen_variable] ** 2) * factor_variances
        total_variance = np.sum(contributions)
        contribution_pct = (contributions / total_variance) * 100 if total_variance > 0 else np.zeros_like(
            contributions)

        plt.figure(figsize=(10, 6))
        plt.bar(range(self.model.q), contribution_pct, color='skyblue', edgecolor='black')
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('Contribution to Variance (%)', fontsize=12)
        plt.title(f'Factor Contributions to {self.model.df.columns[chosen_variable]} Variance (Varimax)', fontsize=14)
        plt.xticks(range(self.model.q), [f'Factor {i + 1}' for i in range(self.model.q)])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_factors_time_series(self):
        """Plot the time series of all factors starting from January 2000."""
        mask = self.model.dates[:-self.model.h] >= '2000-01-01'
        plot_dates = self.model.dates[:-self.model.h][mask]
        plot_factors = self.model.factors[mask, :]

        plt.figure(figsize=(30, 20))
        colors = plt.cm.tab10(np.linspace(0, 1, self.model.q))
        for i in range(self.model.q):
            plt.plot(plot_dates, plot_factors[:, i], label=f'Factor {i + 1}', color=colors[i], linewidth=1.5)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Factor Value', fontsize=12)
        plt.title('Time Series of Extracted Factors (from January 2000)', fontsize=16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def plot_common_component_r2(self, true_values, common_components):
        """
        Plot a bar graph of R² for each variable, showing variance explained by the common component.

        Parameters:
        - true_values: np.array, actual data (T, N)
        - common_components: np.array, common components from the model (T, N)
        """
        if true_values.shape != common_components.shape:
            raise ValueError("Shape mismatch between true_values and common_components.")

        # Initialize arrays
        n_vars = true_values.shape[1]
        r2_values = np.zeros(n_vars)

        # Calculate R² for each variable
        for i in range(n_vars):
            true_var = true_values[:, i]
            common_var = common_components[:, i]
            ss_tot = np.sum((true_var - np.mean(true_var)) ** 2)
            ss_res = np.sum((true_var - common_var) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_values[i] = r2 * 100  # Convert to percentage

        # Plotting
        plt.figure(figsize=(40, 20))
        plt.bar(self.model.df.columns, r2_values, color='teal', edgecolor='black')
        plt.xlabel('Variables', fontsize=20)
        plt.ylabel('Explained Variance % (R²)', fontsize=20)
        plt.title('Variance Explained by Common Component per Variable', fontsize=30)
        plt.xticks(rotation=90, fontsize=15)
        plt.ylim(0, 100)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_common_component_r2_with_varimax(self, true_values, common_components):
        """
        Plot a stacked bar graph of R² for each variable, showing variance explained by each factor
        within the common component, comparing original PCA and Varimax rotation. Ensures total R²
        (bar heights) are identical, with only factor contributions differing.

        Parameters:
        - true_values: np.array, actual data (T, N)
        - common_components: np.array, common components from the model (T, N)
        """
        if true_values.shape != common_components.shape:
            raise ValueError("Shape mismatch between true_values and common_components.")
        if self.model.pca is None:
            raise ValueError("PCA model not fitted. Run fit_static_model() first.")

        # Initialize parameters
        n_vars = true_values.shape[1]
        n_factors = self.model.q
        r2_values = np.zeros(n_vars)  # Shared R² for both original and Varimax
        factor_contributions_original = np.zeros((n_factors, n_vars))
        factor_contributions_varimax = np.zeros((n_factors, n_vars))

        # Calculate total R² using original common component (same for both)
        for i in range(n_vars):
            true_var = true_values[:, i]
            common_var = common_components[:, i]
            ss_tot = np.sum((true_var - np.mean(true_var)) ** 2)
            ss_res = np.sum((true_var - common_var) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_values[i] = r2 * 100  # Convert to percentage

        # Get original loadings and factors
        loadings = self.model.pca.components_.T  # Shape: (n_features, n_factors)
        factors = self.model.factors  # Shape: (n_samples, n_factors)

        # Original PCA: Compute factor contributions
        factor_variances = np.var(factors, axis=0)  # Shape: (n_factors,)
        loadings_squared = loadings ** 2  # Shape: (n_features, n_factors)
        total_variance = np.var(true_values, axis=0)  # Shape: (n_features,)
        contributions_sum = np.zeros(n_vars)
        for j in range(n_factors):
            # Variance explained by factor j for each variable
            factor_contributions_original[j] = (loadings_squared[:, j] * factor_variances[j]) / total_variance * 100
            contributions_sum += factor_contributions_original[j]
        # Normalize contributions to match total R²
        for j in range(n_factors):
            factor_contributions_original[j] *= r2_values / contributions_sum

        # Varimax rotation
        rotator = Rotator(method='varimax')
        rotated_loadings = rotator.fit_transform(loadings)  # Shape: (n_features, n_factors)
        # Compute rotated factors: rotated_factors = X_train @ rotated_loadings
        rotated_factors = self.model.X_train @ rotated_loadings  # Shape: (n_samples, n_factors)
        rotated_factor_variances = np.var(rotated_factors, axis=0)  # Shape: (n_factors,)
        rotated_loadings_squared = rotated_loadings ** 2  # Shape: (n_features, n_factors)
        contributions_sum = np.zeros(n_vars)
        for j in range(n_factors):
            # Variance explained by factor j for each variable
            factor_contributions_varimax[j] = (rotated_loadings_squared[:, j] * rotated_factor_variances[
                j]) / total_variance * 100
            contributions_sum += factor_contributions_varimax[j]
        # Normalize contributions to match total R²
        for j in range(n_factors):
            factor_contributions_varimax[j] *= r2_values / contributions_sum

        # Verify Varimax common component (for debugging)
        common_varimax = rotated_factors @ rotated_loadings.T  # Shape: (n_samples, n_features)
        r2_varimax_check = np.zeros(n_vars)
        for i in range(n_vars):
            true_var = true_values[:, i]
            common_var = common_varimax[:, i]
            ss_tot = np.sum((true_var - np.mean(true_var)) ** 2)
            ss_res = np.sum((true_var - common_var) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_varimax_check[i] = r2 * 100
        max_diff = np.max(np.abs(r2_values - r2_varimax_check))
        print(f"Maximum difference in R² between original and Varimax: {max_diff:.6f}%")

        # Plotting
        plt.figure(figsize=(16, 8))
        x = np.arange(n_vars)
        width = 0.35  # Width of bars
        colors = plt.cm.tab10(np.linspace(0, 1, n_factors))  # Distinct colors for factors

        # Original PCA: Stacked bars
        bottom_original = np.zeros(n_vars)
        for j in range(n_factors):
            plt.bar(x - width / 2, factor_contributions_original[j], width, bottom=bottom_original,
                    color=colors[j], edgecolor='black', label=f'Factor {j + 1} (Original)' if j == 0 else "")
            bottom_original += factor_contributions_original[j]

        # Varimax: Stacked bars
        bottom_varimax = np.zeros(n_vars)
        for j in range(n_factors):
            plt.bar(x + width / 2, factor_contributions_varimax[j], width, bottom=bottom_varimax,
                    color=colors[j], edgecolor='black', alpha=0.7,
                    label=f'Factor {j + 1} (Varimax)' if j == 0 else "")
            bottom_varimax += factor_contributions_varimax[j]

        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Variance Explained (R², %)', fontsize=12)
        plt.title('Variance Explained per Variable by Factor (Original PCA vs Varimax)', fontsize=14)
        plt.xticks(x, self.model.df.columns, rotation=90, fontsize=8)
        plt.ylim(0, 100)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        # Custom legend to show factor colors once
        legend_labels = [f'Factor {j + 1}' for j in range(n_factors)]
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j]) for j in range(n_factors)]
        plt.legend(legend_handles, legend_labels, title='Factors', fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def plot_variance_explained_per_variable(self, varimax=True):
        """
        Plot variance explained by each factor per variable, with optional Varimax rotation.

        Parameters:
        - varimax: bool, whether to apply Varimax rotation (default: True)
        """
        if self.model.pca is None:
            raise ValueError("Run fit_static_model() first to fit the PCA model.")

        # Get PCA loadings
        loadings = self.model.pca.components_.T  # Shape: (n_features, n_factors)
        factors = self.model.factors  # Shape: (n_samples, n_factors)

        if varimax:
            # Apply Varimax rotation
            rotator = Rotator(method='varimax')
            loadings = rotator.fit_transform(loadings)
            # Compute rotated factors (X_train = factors @ loadings.T + error)
            # Since Varimax is orthogonal, rotated_factors = X_train @ loadings
            rotated_factors = self.model.X_train @ loadings
        else:
            rotated_factors = factors

        # Calculate variance explained
        total_variance = np.var(self.model.X_train, axis=0)  # Variance of each variable
        factor_variances = np.var(rotated_factors, axis=0)  # Variance of each (rotated) factor
        loadings_squared = loadings ** 2

        # Compute variance explained by each factor for each variable
        factor_contributions = loadings_squared * factor_variances  # Shape: (n_features, n_factors)
        variance_explained_pct = (factor_contributions.T / total_variance) * 100  # Shape: (n_factors, n_features)

        # Ensure total variance explained does not exceed 100%
        total_explained = np.sum(variance_explained_pct, axis=0)
        if np.any(total_explained > 100):
            # Normalize to the PCA explained variance ratio
            explained_variance_ratio = np.sum(self.model.pca.explained_variance_ratio_)
            variance_explained_pct *= (100 * explained_variance_ratio) / total_explained

        # Plotting
        plt.figure(figsize=(40, 20))
        bottom = np.zeros(self.model.n)
        colors = plt.cm.tab10(np.linspace(0, 1, self.model.q))

        for i in range(self.model.q):
            plt.bar(self.model.df.columns, variance_explained_pct[i], bottom=bottom,
                    color=colors[i], label=f'Factor {i + 1}', edgecolor='black', linewidth=0.5)
            bottom += variance_explained_pct[i]

        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Percentage of Variance Explained (%)', fontsize=12)
        plt.title(f'Variance Explained by Each hampered per Variable {"(Varimax)" if varimax else ""}', fontsize=16)
        plt.xticks(rotation=90, fontsize=8)
        plt.legend(title='Factors', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        plt.tight_layout()
        plt.show()