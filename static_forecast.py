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
    and forecast h steps ahead.

    Parameters:
    - xt: np.array of shape (T, N), observed data where T is time steps and N is number of variables
    - h: int, forecast horizon

    Returns:
    - forecasts: np.array of shape (h, N), h-step-ahead forecasts for each variable
    - D: np.array of shape (N,), estimated AR(1) coefficients for each variable
    """
    T, N = X.shape

    x_mean = np.mean(xt, axis=0)
    xt_centered = xt - x_mean

    # Initialize arrays
    forecasts = np.zeros((h, N))
    D = np.zeros(N)

    # Fit AR(1) model for each variable
    for i in range(N):
        # Extract time series for variable i
        X_i = xt_centered[:, i]

        # Fit AR(1) model with statsmodels
        # trend='n' - no constant (assumes centered data or Z_t handles intercept)
        model = AutoReg(X_i, lags=1, trend='n')
        result = model.fit()

        # Extract AR(1) coefficient (d_i)
        D[i] = result.params[0]  # First parameter is the AR(1) coefficient

        # Generate h-step-ahead forecasts
        # Start forecasting from the last observation (T) up to T+h-1
        forecasts[:, i] = result.predict(start=T, end=T + h - 1, dynamic=True)

    return forecasts, D


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
    Plot the actual data against the common components, including predictions within the same time frame.

    Parameters:
    - actual: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - common_components: np.array of shape (T-h, n), common components of the data, ending h steps before the end of actual
    - forecast: np.array of shape (h, n), forecasted values for h steps within the actual data time frame
    - chosen_component: int, the variable to plot (0-indexed)
    - dates: pd.Index or pd.DatetimeIndex, the dates for plotting
    - h: int, The number of steps ahead for forecasting within the actual data
    """
    i = chosen_component
    actual_data = actual[:, i]
    common_data = common_components[:, i]
    forecast_values = forecast[:, i]
    plot_dates = dates

    # novel method
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_dates, y=actual_data, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=plot_dates[:-h], y=common_data, mode='lines', name='Common Component', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=plot_dates[-h:], y=forecast_values, mode='lines', name='Forecast',
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


def plot_actual_vs_common_components(actual, common_components, forecast, chosen_component, dates, h):
    """
    Plot the actual data against the common components, including predictions within the same time frame.

    Parameters:
    - actual: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - common_components: np.array of shape (T-h, n), common components of the data, ending h steps before the end of actual
    - forecast: np.array of shape (h, n), forecasted values for h steps within the actual data time frame
    - chosen_component: int, the variable to plot (0-indexed)
    - dates: pd.Index or pd.DatetimeIndex, the dates for plotting
    - h: int, The number of steps ahead for forecasting within the actual data
    """
    i = chosen_component
    actual_data = actual[:, i]
    common_data = common_components[:, i]
    forecast_values = forecast[:, i]
    plot_dates = dates

    # Create a new figure with a large size for better visibility
    plt.figure(figsize=(30, 20))

    # Plot the actual data
    plt.plot(plot_dates, actual_data, label='Actual', color='blue', alpha=0.7)

    # Plot the common components (up to h steps before the end of actual data)
    plt.plot(plot_dates[:-h], common_data, label='Common Component', color='green', alpha=0.7)

    # Plot the forecast
    if h <= 1:
        plt.scatter(plot_dates[-h:], forecast_values, label='Forecast', color='red', alpha=1.0)
    else:
        plt.plot(plot_dates[-h:], forecast_values, label='Forecast', color='red', alpha=0.5)

    # Labeling
    plt.xlabel('Date')
    plt.ylabel(f'Variable {i + 1}')
    plt.title(f'Actual Data vs Common Components with Forecast for Variable {i + 1}')

    # Adjust the legend
    plt.legend()

    # Format x-axis dates
    plt.gcf().autofmt_xdate()

    # Tight layout
    plt.tight_layout()
    plt.show()


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

q = 7  # The number of factors
h = 12  # Horizon - Forecasting step(s) ahead
X_train, X_actual = X[:-h], X[-h:]  # Split data for training and testing

# checks
if h <= 0:
    raise ValueError("Forecast horizon 'h' must be positive.")
if q > min(n, T):
    raise ValueError("Number of factors 'q' cannot exceed min(n, T).")

forecast, common = static_forecast(X_train, h, q)
baseline_forecast, _ = fit_ar1_baseline_statsmodels(X, h)

print("forecast:")
performance(X_actual, forecast)
print("baseline_forecast:")
performance(X_actual, baseline_forecast)

# new_plot_actual_vs_common_components(X, common, baseline_forecast, chosen_component=10, dates=dates, h=h)

new_plot_actual_vs_common_components(X, common, forecast, chosen_component=10, dates=dates, h=h)
