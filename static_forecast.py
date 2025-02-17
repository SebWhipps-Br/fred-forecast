import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


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


def static_factor_direct_forecast(xt, h, q):
    """
    Implement direct forecasting from a static factor model.

    Parameters:
    - xt: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - h: int, forecast horizon
    - q: int, number of factors to consider

    Returns:
    - forecast: np.array of shape (n,), the h-step-ahead forecast for each variable
    """
    T, n = xt.shape
    # Compute the mean of the series
    x_mean = np.mean(xt, axis=0)
    xt_centered = xt - x_mean

    pca = PCA(n_components=q)

    factors = pca.fit_transform(xt_centered)

    common_components = pca.inverse_transform(factors)

    V_x = pca.components_.T  # Now this is n x q

    # Compute Γ_x(-h) - h lags autocovariance matrix
    Γ_x_h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if T > h:
                Γ_x_h[i, j] = np.cov(xt_centered[:-h, i], xt_centered[h:, j])[0, 1]
            else:
                Γ_x_h[i, j] = 0

    # Computes V_x0 Γ_x0 V_x
    V_x0_Γ_x0_V_x = V_x.T @ Γ_x_h @ V_x
    V_x0_inv = np.linalg.inv(V_x0_Γ_x0_V_x)

    # Forecast calculation
    B_OLS = Γ_x_h @ V_x @ V_x0_inv @ V_x.T
    alpha_OLS = x_mean  # Since we center the data, this is essentially zero or mean if not centered

    forecast = alpha_OLS + B_OLS @ (xt[-1] - x_mean)
    return forecast, common_components


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

def multi_plot_forecast_vs_actual(xt, forecast, common_components, h=1):
    """
    Plot the forecast value against the actual value for each variable in the time series.

    Parameters:
    - xt: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - forecast: np.array, forecast of common components for the last time step
    - h: int, The number of steps ahead for forecasting (default is 1)    Returns:
    - None, but displays a plot
    """
    T, n = xt.shape

    actual_last = xt[-1]

    graph_number = 10
    fig, axs = plt.subplots(graph_number, 1, figsize=(10, 5 * graph_number), sharex=True)
    if graph_number == 1:
        axs = [axs]  # Makes sure axs is iterable even if only one subplot

    for i, ax in enumerate(axs):
        # Plot the actual data
        ax.plot(range(T), xt[:, i], label='Actual', color='blue', alpha=0.7)

        # Plot the common components
        ax.plot(range(T), common_components[:, i], label='Common Component', color='green', alpha=0.7)

        # Plot the last actual point
        ax.scatter(T - 1, actual_last[i], color='blue', s=50, zorder=5, label='Last Actual' if i == 0 else "")

        # Plot the forecast
        ax.scatter(T, forecast[i], color='red', label='Forecast' if i == 0 else "", s=50, zorder=5)

        ax.set_ylabel(f'Variable {i + 1}')
        ax.legend()

    plt.xlabel('Time')
    plt.suptitle('Forecast vs Actual Values with Common Components for Each Variable')
    plt.tight_layout()
    plt.show()

def plot_forecast_vs_actual(xt, forecast, common_components, chosen_component, h=1):
    """
    Plot the forecast value against the actual value for each variable in the time series.

    Parameters:
    - xt: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - forecast: np.array, forecast of common components for the last time step
    - h: int, The number of steps ahead for forecasting (default is 1)
    Returns:
    - None, but displays a plot
    """
    T, n = xt.shape
    actual_last = xt[-1]
    i = chosen_component
    plt.figure(figsize=(100, 30))

    # Plot the actual data
    plt.plot(range(T), xt[:, i], label='Actual', color='blue', alpha=0.7)
    # Plot the common components
    plt.plot(range(T), common_components[:, i], label='Common Component', color='green', alpha=0.7)

    # Plot the last actual point
    plt.scatter(T - 1, actual_last[i], color='blue', s=50, zorder=5, label='Last Actual' if i == 0 else "")

    # Plot the forecast
    plt.scatter(T, forecast[i], color='red', label='Forecast' if i == 0 else "", s=50, zorder=5)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel(f'Variable {i + 1}')
    plt.suptitle('Forecast vs Actual Values with Common Components for Each Variable')
    plt.tight_layout()
    plt.show()





# Data loading and preparation
df = pd.read_csv('preprocessed_current.csv', index_col=0, parse_dates=True)
X = df.values

T, n = X.shape
print("T:", T, "n:", n)

# Split data for training and testing
h = 1  # Forecasting one step ahead
X_train, X_actual = X[:-h], X[-h]

# Calculate BIC and AIC
max_factors = 20



icp1, icp2, icp3 = bai_ng_criteria(X_train, max_factors)
plot_bai_ng_criteria(X, max_factors)

# Use BIC for forecasting
q = max_factors


forecast, common = static_factor_direct_forecast(X_train, h, q)
# print("Forecast for h steps ahead:", forecast2)

# Evaluate performance
mse = mean_squared_error(X_actual, forecast)
print(f'Overall Mean Squared Error: {mse}')

# Individual MSE for variables
for i, col in enumerate(df.columns):
    mse_i = mean_squared_error([X_actual[i]], [forecast[i]])
    print(f'MSE for variable {col}: {mse_i}')

plot_forecast_vs_actual(X_train, forecast, common, 10, h=h)
multi_plot_forecast_vs_actual(X_train, forecast, common, h=h)
