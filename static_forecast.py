import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def bai_ng_criteria(X, max_factors):
    """
    Compute Bai & Ng Information Criteria (IC_p1, IC_p2, IC_p3) for factor models with PCA.

    Parameters:
    - X: Numpy array of shape (T, N) where T is time periods and N is number of series
    - max_factors: Integer, maximum number of factors to consider

    Returns:
    - Tuple containing lists of IC_p1, IC_p2, IC_p3 values for each number of factors from 1 to max_factors
    """
    T, n = X.shape
    IC_p1_values, IC_p2_values, IC_p3_values = [], [], []

    for k in range(1, max_factors + 1):
        # Perform PCA to get F and V
        pca = PCA(n_components=k)
        F = pca.fit_transform(X)  # Common factors
        V = pca.components_.T  # Factor loadings

        # Compute residuals
        residuals = X - np.dot(F, V.T)

        # Sum of squared residuals
        ln_V_k_F = np.log(np.sum(residuals ** 2) / (n * T))

        # Compute the criteria
        term1 = np.log((n * T)/(n + T))
        term2 = (n + T)/(n * T)
        C_nT = min(n**0.5, T**0.5)
        term3 = np.log(C_nT**2) / C_nT**2

        # IC_p1
        IC_p1 = ln_V_k_F + k * term1 * term2
        IC_p1_values.append(IC_p1)

        # IC_p2
        IC_p2 = ln_V_k_F + k * term2
        IC_p2_values.append(IC_p2)

        # IC_p3
        IC_p3 = ln_V_k_F + k * term3
        IC_p3_values.append(IC_p3)

    return IC_p1_values, IC_p2_values, IC_p3_values

def er_test(eigenvalues):
    ratios = [eigenvalues[i] / eigenvalues[i + 1] for i in range(len(eigenvalues) - 1)]

    # Looks for the largest drop in the ratio
    drops = [ratios[i] / ratios[i - 1] if i > 0 else 0 for i in range(1, len(ratios))]
    max_drop_index = np.argmax(drops)     # Finds where the maximum drop is

    # The number of factors is where we see the largest drop
    return max_drop_index + 1  # +1 because we started counting from 0


def gr_test(eigenvalues):
    # Computes growth rates
    growth_rates = [eigenvalues[i] / eigenvalues[i + 1] for i in range(len(eigenvalues) - 1)]

    # Computes the ratio of growth rates
    gr_ratios = [growth_rates[i] / growth_rates[i - 1] if i > 0 else 0 for i in range(1, len(growth_rates))]
    # Find where the GR ratio drops significantly
    max_gr_drop_index = np.argmin(gr_ratios) # Arbitrary threshold

    return max_gr_drop_index + 1

def plot_bai_ng_criteria(X, max_factors):
    """
    Compute and plot Bai & Ng Information Criteria (IC_p1, IC_p2, IC_p3) for factor models.

    Parameters:
    - X: Numpy array of shape (T, N) where T is time periods and N is number of series
    - max_factors: Integer, maximum number of factors to consider

    Returns:
    - None, but displays a plot of the criteria vs. number of factors
    """
    IC_p1_values, IC_p2_values, IC_p3_values = bai_ng_criteria(X, max_factors)

    # Number of factors considered
    factors = list(range(1, max_factors + 1))

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(factors, IC_p1_values, label='IC_p1', marker='o')
    plt.plot(factors, IC_p2_values, label='IC_p2', marker='s')
    plt.plot(factors, IC_p3_values, label='IC_p3', marker='^')

    plt.xlabel('Number of Factors')
    plt.ylabel('Information Criteria')
    plt.title('Bai & Ng Information Criteria vs Number of Factors')
    plt.legend()
    plt.grid(True)
    plt.xticks(factors)

    plt.show()

def autocovariance(X, horizon):
    if horizon > T:
        return np.zeros((n, n))
    X_lagged = X[:-horizon]
    X_future = X[horizon:]
    return np.cov(X_lagged.T, X_future.T)[0:n, n:]

def static_factor_direct_forecast(xt, h, q):
    """
    Implement direct forecasting from a static factor model for any forecast horizon h.

    Parameters:
    - xt: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - h: int, forecast horizon
    - q: int, number of factors to consider

    Returns:
    - forecast: np.array of shape (h, n), the h-step-ahead forecasts for each variable, directly computed
    - common_components: np.array, common components of the data
    """
    T, n = xt.shape

    # Compute the mean of the series
    x_mean = np.mean(xt, axis=0)
    xt_centered = xt - x_mean

    # PCA for factor estimation
    pca = PCA(n_components=q)
    factors = pca.fit_transform(xt_centered)
    common_components = pca.inverse_transform(factors)

    # Factor loadings
    V_x = pca.components_.T  # Now this is n x q

    # Initialize forecasts for each horizon
    forecasts = np.zeros((h, n))

    for horizon in range(1, h + 1):  # Start from 1 to h
        # Compute Γ_x(-horizon) - for each horizon
        Γ_x_h = autocovariance(xt_centered, horizon)

        V_x0_inv = np.linalg.pinv(V_x.T @ Γ_x_h @ V_x) # pinv instead of inv for where matrix is near singular

        # Forecast calculation for this horizon
        B_OLS = Γ_x_h @ V_x @ V_x0_inv @ V_x.T

        # Direct forecasting for each horizon
        forecasts[horizon - 1] = x_mean + B_OLS @ (xt[-1] - x_mean)

    if forecasts.ndim == 1:  # If forecast is for one step ahead, adjust to 2D
        forecasts = forecast.reshape(1, -1)

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

    #novel method
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

# Data loading and preparation
df = pd.read_csv('preprocessed_current.csv', index_col=0, parse_dates=True)
X = df.values
dates = df.index # Extract the index which contains the dates




T, n = X.shape
print("T:", T, "n:", n)

# Split data for training and testing
h = 12  # Forecasting one step ahead
X_train, X_actual = X[:-h], X[-h]

# Calculate BIC and AIC
max_factors = 3



icp1, icp2, icp3 = bai_ng_criteria(X_train, max_factors)
# plot_bai_ng_criteria(X, max_factors)

# Use BIC for forecasting
q = max_factors

# checks
if h <= 0:
    raise ValueError("Forecast horizon 'h' must be positive.")
if q > min(n, T):
    raise ValueError("Number of factors 'q' cannot exceed min(n, T).")

forecast, common = static_factor_direct_forecast(X_train, h, q)
print("Forecast for h steps ahead:", forecast.shape)
'''
# Evaluate performance
mse = mean_squared_error(X_actual, forecast)
print(f'Overall Mean Squared Error: {mse}')

# Individual MSE for variables
for i, col in enumerate(df.columns):
    mse_i = mean_squared_error([X_actual[i]], [forecast[i]])
    print(f'MSE for variable {col}: {mse_i}')
'''
plot_actual_vs_common_components(X, common, forecast, chosen_component=10, dates=dates, h=h)
new_plot_actual_vs_common_components(X, common, forecast, chosen_component=10, dates=dates, h=h)