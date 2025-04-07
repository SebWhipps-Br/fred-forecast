import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestRegressor


class StaticFactorModel:
    def __init__(self, filepath, q=7, h=12, pca_type='standard', sparse_alpha=0.1, forecast_method='linear'):
        """
        Initialise the Static Factor Model.

        Parameters:
        - filepath: str, path to the preprocessed CSV file
        - q: int, number of factors
        - h: int, forecast horizon
        - pca_type: str, 'standard' for PCA or 'sparse' for SparsePCA
        - sparse_alpha: float, sparsity penalty for SparsePCA
        - forecast_method: str, 'linear' for LinearRegression or 'rf' for RandomForestRegressor
        """
        self.filepath = filepath
        self.q = q
        self.h = h
        self.pca_type = pca_type
        self.sparse_alpha = sparse_alpha
        self.forecast_method = forecast_method
        self.scaler = StandardScaler()
        self.df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.X = self.scaler.fit_transform(self.df)
        self.dates = self.df.index
        self.T, self.n = self.X.shape
        self.X_train = self.X[:-self.h]
        self.X_actual = self.X[-self.h:]
        if self.h <= 0:
            raise ValueError("Forecast horizon 'h' must be positive.")
        if self.q > min(self.n, self.T):
            raise ValueError("Number of factors 'q' cannot exceed min(n, T).")
        self.forecast = None
        self.common = None
        self.baseline_forecast = None
        self.fitted_values = None
        self.loadings = None
        self.pca = None
        self.factors = None

    def fit_ar1_baseline(self):
        """Fit AR(1) baseline model and forecast."""
        T, N = self.X.shape
        x_mean = np.mean(self.X, axis=0)
        xt_centered = self.X - x_mean

        forecasts = np.zeros((self.h, N))
        D = np.zeros(N)
        fitted_values = np.zeros((T, N))

        for i in range(N):
            X_i = xt_centered[:, i]
            model = AutoReg(X_i, lags=1, trend='n')
            result = model.fit()
            D[i] = result.params[0]
            forecasts[:, i] = result.predict(start=T, end=T + self.h - 1, dynamic=True)
            fitted_values[1:, i] = D[i] * X_i[:-1]
            fitted_values[0, i] = X_i[0]

        fitted_values += x_mean
        self.baseline_forecast, self.fitted_values = forecasts, fitted_values

    def fit_static_model(self):
        """Fit the static factor model, extracting factors and computing common components."""
        T, N = self.X_train.shape

        # Select PCA type and fit
        if self.pca_type == 'sparse':
            self.pca = SparsePCA(n_components=self.q, alpha=self.sparse_alpha, random_state=42)
        else:
            self.pca = PCA(n_components=self.q)

        self.factors = self.pca.fit_transform(self.X_train)
        self.common = self.pca.inverse_transform(self.factors)
        self.loadings = self.pca.components_.T

    def forecast_static(self):
        """Fit forecasting models over horizons and generate forecasts using the static factor model."""
        if self.factors is None or self.pca is None:
            raise ValueError("Must run fit_static_model() before forecasting.")

        T, N = self.X_train.shape
        X_forecast = np.zeros((self.h, N))

        # Choose forecasting model based on forecast_method
        if self.forecast_method == 'rf':
            model_class = RandomForestRegressor(n_estimators=100, random_state=123)
        elif self.forecast_method == 'linear':
            model_class = LinearRegression()
        else:
            raise ValueError("forecast_method must be 'linear' or 'rf'")

        # Fit and forecast for each horizon
        for horizon in range(1, self.h + 1):
            X_target = self.X_train[horizon:]
            if len(X_target) == 0:
                raise ValueError(f"Not enough data for horizon {horizon}.")
            factors_t = self.factors[:-horizon]
            model = model_class if self.forecast_method == 'linear' else RandomForestRegressor(n_estimators=100, random_state=123)
            model.fit(factors_t, X_target)
            last_factors = self.factors[-1, :].reshape(1, -1)
            X_forecast[horizon - 1] = model.predict(last_factors)

        self.forecast = X_forecast

    def evaluate_performance(self, true_values, predicted_values, model_name=""):
        """Evaluate forecast performance."""
        print(f"{model_name} Performance:")
        print("true_values.shape:", true_values.shape)
        print("predicted_values.shape:", predicted_values.shape)
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        mape = mean_absolute_percentage_error(true_values, predicted_values)
        print("mse:     ", mse)
        print("mae:     ", mae)
        print("mape:    ", mape)
        print()

    def plot_actual_vs_common_plotly(self, actual, common_components, forecast, chosen_component):
        """Plot using Plotly."""
        i = chosen_component
        actual_data = actual[:, i]
        common_data = common_components[:, i]
        forecast_values = forecast[:, i]
        plot_dates = self.dates

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_dates, y=actual_data, mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=plot_dates[:-self.h], y=common_data, mode='lines', name='Common Component',
                                 line=dict(color='green')))
        fig.add_trace(go.Scatter(x=plot_dates[-self.h:], y=forecast_values, mode='lines', name='Forecast',
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

    def plot_actual_vs_common_matplotlib(self, actual, common_components, forecast, chosen_component):
        """Plot using Matplotlib with solid red forecast line."""
        i = chosen_component
        actual_data = actual[:, i]
        common_data = common_components[:, i]
        forecast_values = forecast[:, i]
        plot_dates = self.dates

        plt.figure(figsize=(30, 20))
        plt.plot(plot_dates, actual_data, label='Actual', color='blue', alpha=0.7)
        plt.plot(plot_dates[:-self.h], common_data, label='Common Component', color='green', alpha=0.7)
        plt.plot(plot_dates[-self.h:], forecast_values, label='Forecast', color='red', linestyle='-', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel(f'Variable {i + 1}')
        plt.title(f'Actual Data vs Common Components with Forecast for Variable {i + 1}')
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def plot_highres(self, actual, common_components, forecast, chosen_component,
                     save_path_prefix='figures/forecast_plot'):
        """High-resolution plot for poster."""
        i = chosen_component
        variable_name = self.df.columns[i]
        mask = self.dates >= '2000-01-01'
        plot_dates = self.dates[mask]
        actual_data = actual[mask, i]
        common_data = common_components[mask[:-self.h], i]
        forecast_values = forecast[:, i]

        plt.figure(figsize=(12, 8), dpi=300)
        plt.plot(plot_dates, actual_data, label='Actual', color='blue', linewidth=2)
        plt.plot(plot_dates[:-self.h], common_data, label='Common Component', color='green', linewidth=2)
        plt.plot(plot_dates[-self.h:], forecast_values, label='Forecast', color='red', linestyle='--', linewidth=2)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel(variable_name, fontsize=14)
        plt.title(f'Actual vs Common Components with Forecast for {variable_name}', fontsize=16, pad=10)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        save_path = f'{save_path_prefix}_{variable_name}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_factor_loadings_heatmap(self):
        """Heatmap of factor loadings for all variables."""
        if self.pca is None:
            raise ValueError("Run fit_static_model() first to fit the PCA model.")

        loadings = self.pca.components_.T  # Shape: (N, q), variables x factors
        plt.figure(figsize=(20, 30))
        sns.heatmap(loadings, cmap='coolwarm', center=0,
                    xticklabels=[f'Factor {i + 1}' for i in range(self.q)],
                    yticklabels=self.df.columns,
                    annot=False, fmt='.2f')
        plt.title('Factor Loadings Heatmap', fontsize=16)
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('Variables', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_variable_factor_contribution(self, chosen_variable):
        """Bar plot of factor contributions for a specific variable."""
        if self.pca is None:
            raise ValueError("Run fit_static_model() first to fit the PCA model.")

        loadings = self.pca.components_.T  # Shape: (N, q)
        variable_name = self.df.columns[chosen_variable]
        variable_loadings = loadings[chosen_variable]

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, self.q + 1), variable_loadings, color='skyblue')
        plt.xlabel('Factor', fontsize=12)
        plt.ylabel('Loading', fontsize=12)
        plt.title(f'Factor Contributions to {variable_name}', fontsize=14)
        plt.xticks(range(1, self.q + 1), [f'Factor {i}' for i in range(1, self.q + 1)])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_factors_time_series(self):
        """Plot the time series of all factors starting from January 2000."""
        mask = self.dates[:-self.h] >= '2000-01-01'  # Filter dates from 2000 onwards
        plot_dates = self.dates[:-self.h][mask]
        plot_factors = self.factors[mask, :]  # Filter factors accordingly

        plt.figure(figsize=(30, 20))
        colors = plt.cm.tab10(np.linspace(0, 1, self.q))  # Use a colormap for distinct colors
        for i in range(self.q):
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
        total_variance = np.var(self.X_train, axis=0)  # Shape: (N,)
        factor_variances = np.var(self.factors, axis=0)  # Variance of each factor [q]
        loadings_squared = self.loadings ** 2  # Shape: [N, q]
        factor_contributions = loadings_squared * factor_variances  # Shape: [N, q]
        variance_explained_pct = (factor_contributions.T / total_variance) * 100  # Shape: [q, N]
        variance_explained_pct = np.clip(variance_explained_pct, 0, 100)

        plt.figure(figsize=(40, 20))
        bottom = np.zeros(self.n)
        colors = plt.cm.tab10(np.linspace(0, 1, self.q))

        for i in range(self.q):
            plt.bar(self.df.columns, variance_explained_pct[i], bottom=bottom,
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

    def run(self, chosen_component=71):
        """Run the full pipeline."""
        print("T:", self.T, "n:", self.n)

        # Fit models
        self.fit_static_model()  # Fit PCA and extract factors
        self.forecast_static()   # Fit forecasting models and generate forecasts
        self.fit_ar1_baseline()

        # Print shapes
        print("common.shape:", self.common.shape)
        print("fitted_values[:-h].shape:", self.fitted_values[:-self.h].shape)

        # Evaluate
        self.evaluate_performance(self.X_actual, self.baseline_forecast, "Baseline Forecast")
        self.evaluate_performance(self.X_actual, self.forecast, "Static Forecast")


        # Plot existing visualizations
        # self.plot_actual_vs_common_plotly(self.X, self.fitted_values[:-self.h], self.baseline_forecast, chosen_component)
        # self.plot_actual_vs_common_plotly(self.X, self.common, self.forecast, chosen_component)
        # self.plot_actual_vs_common_matplotlib(self.X, self.common, self.forecast, chosen_component)
        self.plot_highres(self.X, self.common, self.forecast, chosen_component)

        # Factor visualizations
        # self.plot_factor_loadings_heatmap()
        # self.plot_variable_factor_contribution(chosen_component)
        # self.plot_factors_time_series()
        # self.plot_variance_explained_per_variable()


# Usage
if __name__ == "__main__":
    variable_number = 71 # 71 - S&P PE Ratio
    model = StaticFactorModel(filepath='preprocessed_current.csv', q=7, h=12, pca_type='standard', forecast_method='linear')
    model.run(chosen_component=variable_number)

    print("\n~~~~~~~~~~~~~~~~~~~\n")

    model = StaticFactorModel(filepath='preprocessed_current.csv', q=7, h=12, pca_type='standard', forecast_method='rf')
    model.run(chosen_component=variable_number)