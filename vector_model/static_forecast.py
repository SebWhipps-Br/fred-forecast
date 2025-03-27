import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg


def fit_ar1_baseline_statsmodels(xt, h):
    """
    Fit an AR(1) baseline model X_t = D * X_{t-1} + Z_t for each variable using statsmodels,
    and forecast h steps ahead. Also returns fitted values as a pseudo-common component.

    Parameters:
    - xt: np.array of shape (T, N), observed data where T is time steps and N is number of variables
    - h: int, forecast horizon

    Returns:
    - forecasts: np.array of shape (h, N), h-step-ahead forecasts for each variable
    - D: np.array of shape (N,), estimated AR(1) coefficients for each variable
    - fitted_values: np.array of shape (T, N), fitted values (pseudo-common component)
    """
    T, N = xt.shape  # Corrected from X to xt

    x_mean = np.mean(xt, axis=0)
    xt_centered = xt - x_mean

    # Initialize arrays
    forecasts = np.zeros((h, N))
    D = np.zeros(N)
    fitted_values = np.zeros((T, N))  # Shape matches xt for fitted values

    # Fit AR(1) model for each variable
    for i in range(N):
        # Extract time series for variable i
        X_i = xt_centered[:, i]

        # Fit AR(1) model with statsmodels
        model = AutoReg(X_i, lags=1, trend='n')
        result = model.fit()

        # Extract AR(1) coefficient (d_i)
        D[i] = result.params[0]

        # Generate h-step-ahead forecasts
        forecasts[:, i] = result.predict(start=T, end=T + h - 1, dynamic=True)

        # Compute fitted values (X̂_t = d_i * X_{t-1}) for the training period
        fitted_values[1:, i] = D[i] * X_i[:-1]  # Starts at t=1 due to lag
        fitted_values[0, i] = X_i[0]  # First value can’t be fitted, use observed

    # Adjust fitted values to original scale (add mean back)
    fitted_values += x_mean

    return forecasts, D, fitted_values


def static_forecast(X, h, q):
    """
    Adapted Stock and Watson static factor model to forecast all variables in X_{t+h} simultaneously.


    Parameters:
    - X: np.array of shape (T, N), data where T is time steps and N is number of variables
    - h: int, forecast horizon
    - q: int, number of factors to extract via PCA

    Returns:
    - X_forecast: np.array of shape (1, N), h-step-ahead forecast for all variables
    - factors: np.array of shape (T, q), estimated factors
    """
    T, N = X.shape

    # Extract factors using PCA
    pca = PCA(n_components=q)
    factors = pca.fit_transform(X)  # Shape: (T, q)
    common = pca.inverse_transform(factors)

    X_forecast = np.zeros((h, N))  # Initialises forecast array

    # Fit regression and forecast for each horizon
    for horizon in range(1, h + 1):
        # Prepare regression data for X_{t+horizon}
        X_target = X[horizon:]  # Shape: (T-horizon, N)
        if len(X_target) == 0:
            raise ValueError(f"Not enough data for horizon {horizon}.")
        factors_t = factors[:-horizon]  # Shape: (T-horizon, q)

        # Fit multivariate OLS regression: X_{t+horizon} = F_t * B + E
        model = LinearRegression()
        model.fit(factors_t, X_target)

        # Forecast X_{T+horizon} using the last factors
        last_factors = factors[-1, :].reshape(1, -1)  # Shape: (1, q)
        X_forecast[horizon - 1] = model.predict(last_factors)  # Shape: (1, N)

    return X_forecast, common


def new_plot_actual_vs_common_components(actual, common_components, forecast, chosen_component, dates, h):
    """
    Plot the actual data against the common components, including predictions within the same time frame using Plotly.
    """
    i = chosen_component
    actual_data = actual[:, i]
    common_data = common_components[:, i]
    forecast_values = forecast[:, i]
    plot_dates = dates

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_dates, y=actual_data, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=plot_dates[:-h], y=common_data, mode='lines', name='Common Component', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=plot_dates[-h:], y=forecast_values, mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
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

def plot_actual_vs_common_components(actual, common_components, forecast, chosen_component, dates, h):
    """
    Plot the actual data against the common components, including predictions within the same time frame using Matplotlib.
    """
    i = chosen_component
    actual_data = actual[:, i]
    common_data = common_components[:, i]
    forecast_values = forecast[:, i]
    plot_dates = dates

    plt.figure(figsize=(30, 20))
    plt.plot(plot_dates, actual_data, label='Actual', color='blue', alpha=0.7)
    plt.plot(plot_dates[:-h], common_data, label='Common Component', color='green', alpha=0.7)
    if h <= 1:
        plt.scatter(plot_dates[-h:], forecast_values, label='Forecast', color='red', alpha=1.0)
    else:
        plt.plot(plot_dates[-h:], forecast_values, label='Forecast', color='red', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel(f'Variable {i + 1}')
    plt.title(f'Actual Data vs Common Components with Forecast for Variable {i + 1}')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_actual_vs_common_components_highres(actual, common_components, forecast, chosen_component, dates, df, h, save_path_prefix='figures/forecast_plot'):
    """
    High-resolution Matplotlib plot for poster, starting from 2000, with variable name in title.
    Parameters:
    - actual: np.array of shape (T, n), observed data
    - common_components: np.array of shape (T-h, n), common components
    - forecast: np.array of shape (h, n), forecasted values
    - chosen_component: int, variable index
    - dates: pd.DatetimeIndex, dates for plotting
    - df: pd.DataFrame, original data with column names
    - h: int, forecast horizon
    - save_path_prefix: str, prefix for saved file path
    """
    i = chosen_component
    variable_name = df.columns[i]  # Get variable name (e.g., "RPI")
    mask = dates >= '2000-01-01'  # Filter from 2000 onwards
    plot_dates = dates[mask]
    actual_data = actual[mask, i]
    common_data = common_components[mask[:-h], i]  # Adjust for horizon
    forecast_values = forecast[:, i]

    plt.figure(figsize=(12, 8), dpi=300)  # Large, high-res
    plt.plot(plot_dates, actual_data, label='Actual', color='blue', linewidth=2)
    plt.plot(plot_dates[:-h], common_data, label='Common Component', color='green', linewidth=2)
    plt.plot(plot_dates[-h:], forecast_values, label='Forecast', color='red', linestyle='--', linewidth=2)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(variable_name, fontsize=14)
    plt.title(f'Actual vs Common Components with Forecast for {variable_name}', fontsize=16, pad=10)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    # Save high-res plot
    save_path = f'{save_path_prefix}_{variable_name}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def performance(true_values, predicted_values):
    print("true_values.shape:", true_values.shape)
    print("predicted_values.shape:", predicted_values.shape)
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    mape = mean_absolute_percentage_error(true_values, predicted_values)
    print("mse:     ", mse)
    print("mae:     ", mae)
    print("mape:   ", mape)

# Data loading and preparation
df = pd.read_csv('preprocessed_current.csv', index_col=0, parse_dates=True)
scaler = StandardScaler()
X = scaler.fit_transform(df)
dates = df.index

T, n = X.shape
print("T:", T, "n:", n)

q = 7  # Number of factors
h = 12  # Horizon
X_train, X_actual = X[:-h], X[-h:]  # Split data

# Checks
if h <= 0:
    raise ValueError("Forecast horizon 'h' must be positive.")
if q > min(n, T):
    raise ValueError("Number of factors 'q' cannot exceed min(n, T).")

# Run models
forecast, common = static_forecast(X_train, h, q)
baseline_forecast, _, fitted_values = fit_ar1_baseline_statsmodels(X, h)
print(common.shape)
print(fitted_values[:-h].shape)

print("forecast:")
performance(X_actual, forecast)
print("baseline_forecast:")
performance(X_actual, baseline_forecast)

component = 71  # Example: RPI if it’s column 40

# Existing plots
new_plot_actual_vs_common_components(X, fitted_values[:-h], baseline_forecast, chosen_component=component, dates=dates, h=h)
new_plot_actual_vs_common_components(X, common, forecast, chosen_component=component, dates=dates, h=h)

# High-res plot for poster
plot_actual_vs_common_components_highres(X, common, forecast, chosen_component=component, dates=dates, df=df, h=h)
