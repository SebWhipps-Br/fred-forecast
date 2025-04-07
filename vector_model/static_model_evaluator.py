import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from scipy.stats import norm


class ModelEvaluator:
    def __init__(self, model):
        """
        Initialise the ModelEvaluator with a StaticFactorModel instance.

        Parameters:
        - model: StaticFactorModel instance
        """
        self.model = model


    def evaluate_performance(self, true_values, predicted_values, model_name="", baseline_values=None):
        """
        Evaluate forecast performance with metrics aligned with factor model literature.

        Parameters:
        - true_values: np.array, actual values (T_h, N)
        - predicted_values: np.array, forecasted values (T_h, N)
        - model_name: str, name of the model for printing
        - baseline_values: np.array, optional baseline forecast for Diebold-Mariano test (T_h, N)
        """
        print(f"{model_name} Performance:")
        print("true_values.shape:", true_values.shape)
        print("predicted_values.shape:", predicted_values.shape)

        # Standard Metrics
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predicted_values)
        mape = mean_absolute_percentage_error(true_values, predicted_values)

        print("MSE:      ", mse)
        print("RMSE:     ", rmse)
        print("MAE:      ", mae)
        print("MAPE (%): ", mape)

        # Adjusted R²
        n, p = true_values.shape[0], self.model.q  # n: forecast horizon steps, p: number of factors
        ss_tot = np.sum((true_values - np.mean(true_values, axis=0)) ** 2)
        ss_res = np.sum((true_values - predicted_values) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
        print("Adjusted R²:", adj_r2)

        # Directional Accuracy
        true_diff = np.sign(true_values[1:] - true_values[:-1])
        pred_diff = np.sign(predicted_values[1:] - predicted_values[:-1])
        directional_accuracy = np.mean(true_diff == pred_diff) * 100
        print("Directional Accuracy (%):", directional_accuracy)

        # Diebold-Mariano Test (if baseline provided)
        if baseline_values is not None:
            if baseline_values.shape != true_values.shape:
                print("Warning: Baseline shape mismatch, skipping DM test.")
            else:
                # Compute loss differential (mean squared error across variables)
                loss_forecast = np.mean((true_values - predicted_values) ** 2, axis=1)  # Shape: (T_h,)
                loss_baseline = np.mean((true_values - baseline_values) ** 2, axis=1)  # Shape: (T_h,)
                d = loss_forecast - loss_baseline  # 1D array of length T_h

                # DM test statistic (assuming no autocorrelation for simplicity)
                dm_stat = np.mean(d) / np.sqrt(np.var(d) / len(d))
                dm_pvalue = 2 * (1 - norm.cdf(abs(dm_stat)))
                print("Diebold-Mariano Test:")
                print("  DM Statistic:", dm_stat)
                print("  P-value:", dm_pvalue)
                print("  (Positive DM stat favors baseline; negative favors forecast)")
        print()

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
        """Heatmap of factor loadings for all variables."""
        if self.model.pca is None:
            raise ValueError("Run fit_static_model() first to fit the PCA model.")

        loadings = self.model.pca.components_.T
        plt.figure(figsize=(20, 30))
        sns.heatmap(loadings, cmap='coolwarm', center=0,
                    xticklabels=[f'Factor {i + 1}' for i in range(self.model.q)],
                    yticklabels=self.model.df.columns,
                    annot=False, fmt='.2f')
        plt.title('Factor Loadings Heatmap', fontsize=16)
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('Variables', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_variable_factor_contribution(self, chosen_variable):
        """Bar plot of factor contributions for a specific variable."""
        if self.model.pca is None:
            raise ValueError("Run fit_static_model() first to fit the PCA model.")

        loadings = self.model.pca.components_.T
        variable_name = self.model.df.columns[chosen_variable]
        variable_loadings = loadings[chosen_variable]

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, self.model.q + 1), variable_loadings, color='skyblue')
        plt.xlabel('Factor', fontsize=12)
        plt.ylabel('Loading', fontsize=12)
        plt.title(f'Factor Contributions to {variable_name}', fontsize=14)
        plt.xticks(range(1, self.model.q + 1), [f'Factor {i}' for i in range(1, self.model.q + 1)])
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

    def plot_variance_explained_per_variable(self):
        """Plot a stacked bar chart of the percentage of variance explained by each factor for each variable."""
        total_variance = np.var(self.model.X_train, axis=0)
        factor_variances = np.var(self.model.factors, axis=0)
        loadings_squared = self.model.loadings ** 2
        factor_contributions = loadings_squared * factor_variances
        variance_explained_pct = (factor_contributions.T / total_variance) * 100
        variance_explained_pct = np.clip(variance_explained_pct, 0, 100)

        plt.figure(figsize=(40, 20))
        bottom = np.zeros(self.model.n)
        colors = plt.cm.tab10(np.linspace(0, 1, self.model.q))

        for i in range(self.model.q):
            plt.bar(self.model.df.columns, variance_explained_pct[i], bottom=bottom,
                    color=colors[i], label=f'Factor {i + 1}', edgecolor='black', linewidth=0.5)
            bottom += variance_explained_pct[i]

        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Percentage of Variance Explained (%)', fontsize=12)
        plt.title('Variance Explained by Each Factor per Variable', fontsize=16)
        plt.xticks(rotation=90, fontsize=8)
        plt.legend(title='Factors', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        plt.tight_layout()
        plt.show()
