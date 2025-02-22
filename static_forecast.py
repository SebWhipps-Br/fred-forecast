import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


def autocovariance(X, horizon):
    if horizon > T:  # The case where there is no data to compute the horizon
        return np.zeros((n, n))
    X_lagged = X[:-horizon]
    X_future = X[horizon:]
    return np.cov(X_lagged.T, X_future.T)[0:n, n:]


def static_factor_direct_forecast(xt, h, q):
    """
    Implements direct forecasting from a static factor model for any forecast horizon h.

    Parameters:
    - xt: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - h: int, forecast horizon
    - q: int, number of factors to consider

    Returns:
    - forecast: np.array of shape (h, n), the h-step-ahead forecasts for each variable, directly computed
    - common_components: np.array, common components of the data
    """
    # Compute the mean of the series
    x_mean = np.mean(xt, axis=0)
    xt_centered = xt - x_mean

    # PCA for factor estimation
    pca = PCA(n_components=q)
    factors = pca.fit_transform(xt_centered)
    common_components = pca.inverse_transform(factors)

    # Factor loadings or V_x
    loadings = pca.components_.T  # n x q

    # Initialize forecasts for each horizon
    forecasts = np.zeros((h, n))

    # Compute Γ_x_h - for each horizon
    for horizon in range(1, h + 1):  # Start from 1 to h

        Γ_x_h = autocovariance(xt_centered, horizon)

        V_x0_inv = np.linalg.pinv(
            loadings.T @ Γ_x_h @ loadings)  # pinv instead of inv for where matrix is near singular

        # Forecast calculation for this horizon
        B_OLS = Γ_x_h @ loadings @ V_x0_inv @ loadings.T

        # Direct forecasting for each horizon
        forecasts[horizon - 1] = x_mean + B_OLS @ (xt[-1] - x_mean)

    return forecasts, common_components


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

q = 7   # The number of factors
h = 1   # Horizon - Forecasting step(s) ahead
X_train, X_actual = X[:-h], X[-h:]  # Split data for training and testing




# checks
if h <= 0:
    raise ValueError("Forecast horizon 'h' must be positive.")
if q > min(n, T):
    raise ValueError("Number of factors 'q' cannot exceed min(n, T).")

forecast, common = static_factor_direct_forecast(X_train, h, q)

performance(X_actual, forecast)

plot_actual_vs_common_components(X, common, forecast, chosen_component=10, dates=dates, h=h)
new_plot_actual_vs_common_components(X, common, forecast, chosen_component=10, dates=dates, h=h)