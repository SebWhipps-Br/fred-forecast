import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


def calculate_bic_aic(X, max_factors):
    """
    Calculate BIC and AIC for different numbers of factors.

    Parameters:
    - X: np.array of shape (T, n), observed data
    - max_factors: int, maximum number of factors to consider

    Returns:
    - bic_scores: list of BIC scores
    - aic_scores: list of AIC scores
    """
    T, n = X.shape
    bic_scores, aic_scores = [], []

    for q in range(1, max_factors + 1):
        pca = PCA(n_components=q)
        X_reconstructed = pca.fit_transform(X)
        mse = mean_squared_error(X, pca.inverse_transform(X_reconstructed))

        # AIC and BIC calculation
        k = q * n  # Number of parameters
        aic = T * n * np.log(mse) + 2 * k
        bic = T * n * np.log(mse) + k * np.log(T * n)

        aic_scores.append(aic)
        bic_scores.append(bic)

    return bic_scores, aic_scores


def bai_ng_aic(X, max_factors):
    """
    Calculate Bai & Ng's AIC1, AIC2, AIC3 for selecting the number of factors.

    Parameters:
    - X: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - max_factors: int, maximum number of factors to consider

    Returns:
    - aic1_scores: list of AIC1 scores
    - aic2_scores: list of AIC2 scores
    - aic3_scores: list of AIC3 scores
    """
    T, n = X.shape
    aic1_scores = []
    aic2_scores = []
    aic3_scores = []

    for r in range(1, max_factors + 1):
        # Fit PCA with r components
        pca = PCA(n_components=r)
        X_hat = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_hat)

        # Calculate residuals
        residuals = X - X_reconstructed
        sigma_e_squared = np.mean(residuals ** 2)  # Residual variance

        # Calculate penalty terms
        k1 = (n + T) / (n * T)
        k2 = k1 * np.log(min(n, T))
        k3 = k1 * np.log(n * T / (n + T))

        # Calculate AIC scores
        aic1 = np.log(sigma_e_squared / (n * T)) + (r * (n + T - r) / (n * T)) * k1
        aic2 = np.log(sigma_e_squared / (n * T)) + (r * (n + T - r) / (n * T)) * k2
        aic3 = np.log(sigma_e_squared / (n * T)) + (r * (n + T - r) / (n * T)) * k3

        aic1_scores.append(aic1)
        aic2_scores.append(aic2)
        aic3_scores.append(aic3)

    return aic1_scores, aic2_scores, aic3_scores


def bai_ng_bic(X, max_factors):
    """
    Calculate Bai & Ng's BIC1, BIC2, BIC3 for selecting the number of factors.

    Parameters:
    - X: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - max_factors: int, maximum number of factors to consider

    Returns:
    - bic1_scores: list of BIC1 scores
    - bic2_scores: list of BIC2 scores
    - bic3_scores: list of BIC3 scores
    """
    T, n = X.shape
    bic1_scores = []
    bic2_scores = []
    bic3_scores = []

    for r in range(1, max_factors + 1):
        # Fit PCA with r components
        pca = PCA(n_components=r)
        X_hat = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_hat)

        # Calculate residuals
        residuals = X - X_reconstructed
        sigma_e_squared = np.mean(residuals ** 2)  # Residual variance

        # Calculate penalty terms
        k1 = np.log((n * T) / (n + T))
        k2 = k1 * ((n + T) / (n * T))
        k3 = k2 * np.log(min(n, T))

        # Calculate BIC scores
        bic1 = np.log(sigma_e_squared / (n * T)) + (r * (n + T - r) / (n * T)) * k1
        bic2 = np.log(sigma_e_squared / (n * T)) + (r * (n + T - r) / (n * T)) * k2
        bic3 = np.log(sigma_e_squared / (n * T)) + (r * (n + T - r) / (n * T)) * k3

        bic1_scores.append(bic1)
        bic2_scores.append(bic2)
        bic3_scores.append(bic3)

    return bic1_scores, bic2_scores, bic3_scores


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


def forecast_common_components(xt, q, h=1):
    """
    Perform static forecasting of common components using PCA.

    Parameters:
    - xt: np.array of shape (T, n), observed data where T is time steps and n is number of variables
    - q: int, number of factors to consider
    - h: int, The lags
    Returns:
    - forecast: np.array, forecast of common components for the last time step
    """
    pca = PCA(n_components=q)
    factors = pca.fit_transform(xt)

    common_components = pca.inverse_transform(factors)

    # P^(nT) - normalized eigenvectors corresponding to the q largest eigenvalues
    P = pca.components_.T  # n x q
    Γ_χ = np.cov(common_components.T) # The covariance matrix of the common component

    # Computes lagged covariance matrix
    Γ_χ_h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if T > h:
                Γ_χ_h[i, j] = np.cov(common_components[:-h, i], common_components[h:, j])[0, 1]
            else:
                Γ_χ_h[i, j] = 0

    # Forecast calculation
    inv_term = np.linalg.inv(P.T @ Γ_χ @ P)
    forecast = (Γ_χ_h @ P @ inv_term) @ P.T @ xt[-1]
    return forecast, common_components


def plot(bic_scores, aic_scores, optimal_q_bic, optimal_q_aic):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_factors + 1), bic_scores, label='BIC', marker='o')
    plt.plot(range(1, max_factors + 1), aic_scores, label='AIC', marker='s')
    plt.xlabel('Number of Factors')
    plt.ylabel('Criterion Value')
    plt.title('BIC and AIC for Different Numbers of Factors')
    plt.legend()
    # Mark minimum points
    plt.plot(optimal_q_bic, bic_scores[optimal_q_bic - 1], 'ro', markersize=10, label=f'Min BIC at {optimal_q_bic}')
    plt.plot(optimal_q_aic, aic_scores[optimal_q_aic - 1], 'gs', markersize=10, label=f'Min AIC at {optimal_q_aic}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_forecast_vs_actual(xt, forecast, common_components, h=1):
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
    if n == 1:
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

def cumulative_variance_explained(explained_variance_ratio, k):
    return np.sum(explained_variance_ratio[:k]) * 100




# Data loading and preparation
df = pd.read_csv('preprocessed_current.csv', index_col=0, parse_dates=True)
X = df.values

T, n = X.shape
print("T:", T, "n:", n)

# Split data for training and testing
h = 1  # Forecasting one step ahead
X_train, X_actual = X[:-h], X[-h]

# Calculate BIC and AIC
max_factors = 50
bic_scores, aic_scores = calculate_bic_aic(X_train, max_factors)

optimal_q_bic = np.argmin(bic_scores) + 1
optimal_q_aic = np.argmin(aic_scores) + 1

aic1, aic2, aic3 = bai_ng_aic(X_train, max_factors)
bic1, bic2, bic3 = bai_ng_bic(X_train, max_factors)
print("aic_1:", aic1, "aic_2:", aic2, "aic_3:", aic3)
print("bic_1", bic1, "bic_2:", bic2, "bic_3:", bic3)


plot(bic_scores, aic_scores, optimal_q_bic, optimal_q_aic)

# Use BIC for forecasting
q = optimal_q_bic

# Forecast
forecast, common = forecast_common_components(X_train, q)

print("forecast.shape:",forecast.shape)
print("X_actual:", X_actual.shape)
# Evaluate performance
mse = mean_squared_error(X_actual, forecast)
print(f'Overall Mean Squared Error: {mse}')

# Individual MSE for variables
for i, col in enumerate(df.columns):
    mse_i = mean_squared_error([X_actual[i]], [forecast[i]])
    print(f'MSE for variable {col}: {mse_i}')

plot_forecast_vs_actual(X_train, forecast, common, h=h)
