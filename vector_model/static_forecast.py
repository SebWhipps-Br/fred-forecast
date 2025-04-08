import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from autoregression_model import AutoRegressionModel  # Import the new class
from static_model_evaluator import ModelEvaluator


class StaticFactorModel:
    def __init__(self, filepath, q=7, h=12, pca_type='standard', sparse_alpha=0.1, forecast_method='linear'):
        """
        Initialize the Static Factor Model.

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
        self.loadings = None
        self.pca = None
        self.factors = None

    def fit_static_model(self):
        """Fit the static factor model, extracting factors and computing common components."""
        T, N = self.X_train.shape

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

        if self.forecast_method == 'rf':
            model_class = RandomForestRegressor(n_estimators=100, random_state=123)
        elif self.forecast_method == 'linear':
            model_class = LinearRegression()
        else:
            raise ValueError("forecast_method must be 'linear' or 'rf'")

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

    def run(self, chosen_component=71):
        """Run the static factor model pipeline."""
        print("T:", self.T, "n:", self.n)
        self.fit_static_model()
        self.forecast_static()
        print("common.shape:", self.common.shape)


if __name__ == "__main__":
    variable_number = 71  # Adjust as needed (e.g., 47 or 71 for S&P PE Ratio)
    HORIZON = 12  # in months

    # Run the AR(1) baseline model and evaluate its performance
    df = pd.read_csv('preprocessed_current.csv', index_col=0, parse_dates=True)
    X = StandardScaler().fit_transform(df)
    ar_model = AutoRegressionModel(X, h=HORIZON)
    ar_model.fit()
    baseline_forecast = ar_model.forecasts
    fitted_values = ar_model.fitted_values
    baseline_evaluator = ModelEvaluator(StaticFactorModel(filepath='preprocessed_current.csv', q=7, h=HORIZON))
    # Evaluate fit
    baseline_evaluator.evaluate_fit(ar_model.X_train, fitted_values, "Baseline AR(1) Fit")  # Use X_train, not X
    baseline_evaluator.evaluate_fit(ar_model.X_train, fitted_values, "Baseline AR(1) Fit", variable_number)
    # Evaluate forecast
    baseline_evaluator.evaluate_performance(baseline_evaluator.model.X_actual, baseline_forecast, "Baseline Forecast (AR(1))")
    baseline_evaluator.evaluate_variable_performance(baseline_evaluator.model.X_actual, baseline_forecast, variable_number, "Baseline Forecast (AR(1))")

    print("\n~~~~~~~~~~~~~~~~~~~\n")

    # Run Static Factor Model with Linear Regression
    model_linear = StaticFactorModel(filepath='preprocessed_current.csv', q=7, h=HORIZON, pca_type='standard', forecast_method='linear')
    model_linear.run(chosen_component=variable_number)
    evaluator_linear = ModelEvaluator(model_linear)
    # Evaluate fit
    evaluator_linear.evaluate_fit(model_linear.X_train, model_linear.common, "Static Linear Fit")
    evaluator_linear.evaluate_fit(model_linear.X_train, model_linear.common, "Static Linear Fit", variable_number)
    # Evaluate forecast
    evaluator_linear.evaluate_performance(model_linear.X_actual, model_linear.forecast, "Static Forecast (Linear)", baseline_forecast)
    evaluator_linear.evaluate_variable_performance(model_linear.X_actual, model_linear.forecast, variable_number, "Static Forecast (Linear)", baseline_forecast)
    evaluator_linear.plot_actual_vs_common(model_linear.X, model_linear.common, model_linear.forecast, variable_number)

    print("\n~~~~~~~~~~~~~~~~~~~\n")

    # Run Static Factor Model with Random Forest
    model_rf = StaticFactorModel(filepath='preprocessed_current.csv', q=7, h=HORIZON, pca_type='standard', forecast_method='rf')
    model_rf.run(chosen_component=variable_number)
    evaluator_rf = ModelEvaluator(model_rf)
    # Evaluate fit
    evaluator_rf.evaluate_fit(model_rf.X_train, model_rf.common, "Static RF Fit")
    evaluator_rf.evaluate_fit(model_rf.X_train, model_rf.common, "Static RF Fit", variable_number)
    # Evaluate forecast
    evaluator_rf.evaluate_performance(model_rf.X_actual, model_rf.forecast, "Static Forecast (RF)", baseline_forecast)
    evaluator_rf.evaluate_variable_performance(model_rf.X_actual, model_rf.forecast, variable_number, "Static Forecast (RF)", baseline_forecast)
    evaluator_rf.plot_actual_vs_common(model_rf.X, model_rf.common, model_rf.forecast, variable_number)