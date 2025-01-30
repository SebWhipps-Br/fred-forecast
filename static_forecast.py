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

def forecast_common_components(xt, q):
    """
    Perform static forecasting of common components using PCA.

    Parameters:
    - xt: np.array of shape (T, n), observed data
    - q: int, number of factors to consider

    Returns:
    - forecast: np.array, forecast of common components
    """
    pca = PCA(n_components=q).fit(xt)
    x_last = xt[-1]
    factor_scores = pca.transform(x_last.reshape(1, -1))[0]
    er = er_test(pca.explained_variance_)
    print('er', er)
    for a in pca.explained_variance_:
        print(round(a, 8))
    return pca.inverse_transform(factor_scores.reshape(1, -1)).flatten()

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


# Data loading and preparation
df = pd.read_csv('preprocessed_current.csv', index_col=0, parse_dates=True)
X = df.values

T, n = X.shape

# Split data for training and testing
h = 1  # Forecasting one step ahead
X_train, X_actual = X[:-h], X[-h]

# Calculate BIC and AIC
max_factors = 20
bic_scores, aic_scores = calculate_bic_aic(X_train, max_factors)

# TODO: Find optimal number of factors
optimal_q_bic = np.argmin(bic_scores) + 1
optimal_q_aic = np.argmin(aic_scores) + 1

plot(bic_scores, aic_scores, optimal_q_bic, optimal_q_aic)

# Use BIC for forecasting
q = optimal_q_bic

# Forecast
forecast = forecast_common_components(X_train, q)

# Evaluate performance
mse = mean_squared_error(X_actual, forecast)
print(f'Overall Mean Squared Error: {mse}')

# Individual MSE for variables
for i, col in enumerate(df.columns):
    mse_i = mean_squared_error([X_actual[i]], [forecast[i]])
    print(f'MSE for variable {col}: {mse_i}')

