import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


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

def er_test(ratios):
    # Looks for the largest drop in the ratio
    drops = [ratios[i] / ratios[i - 1] if i > 0 else 0 for i in range(1, len(ratios))]
    max_drop_index = np.argmax(drops)     # Finds where the maximum drop is

    # The number of factors is where we see the largest drop
    return max_drop_index + 1  # +1 because we started counting from 0


def gr_test(eigenvalues):
    # Computes growth rates
    growth_rates = [eigenvalues[i] / eigenvalues[i + 1] for i in range(len(eigenvalues) - 1)]
    print("growth rates: ", growth_rates)

    # Computes the ratio of growth rates
    gr_ratios = [growth_rates[i] / growth_rates[i - 1] if i > 0 else 0 for i in range(1, len(growth_rates))]
    # Find where the GR ratio drops significantly
    max_gr_drop_index = np.argmin(gr_ratios) # Arbitrary threshold

    return max_gr_drop_index + 1

def plot_bai_ng_criteria(X, max_factors):
    """
    Compute and plot Bai & Ng Information Criteria (IC_p1, IC_p2, IC_p3) for factor models,
    highlighting the minimum value for each criterion in a different color.

    Parameters:
    - X: Numpy array of shape (T, N) where T is time periods and N is number of series
    - max_factors: Integer, maximum number of factors to consider

    Returns:
    - None, but displays a plot of the criteria vs. number of factors with minima highlighted
    """
    # Assuming bai_ng_criteria is defined elsewhere
    IC_p1_values, IC_p2_values, IC_p3_values = bai_ng_criteria(X, max_factors)

    # Number of factors considered
    factors = list(range(1, max_factors + 1))

    # Find indices of minimum values for each criterion
    min_ic_p1_idx = np.argmin(IC_p1_values)
    min_ic_p2_idx = np.argmin(IC_p2_values)
    min_ic_p3_idx = np.argmin(IC_p3_values)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot IC_p1 with minimum highlighted in red
    plt.plot(factors, IC_p1_values, label='IC_p1', marker='o', color='blue')
    plt.plot(factors[min_ic_p1_idx], IC_p1_values[min_ic_p1_idx],
             marker='o', color='red', markersize=10, label=f'Min IC_p1')

    # Plot IC_p2 with minimum highlighted in green
    plt.plot(factors, IC_p2_values, label='IC_p2', marker='s', color='orange')
    plt.plot(factors[min_ic_p2_idx], IC_p2_values[min_ic_p2_idx],
             marker='s', color='green', markersize=10, label=f'Min IC_p2')

    # Plot IC_p3 with minimum highlighted in purple
    plt.plot(factors, IC_p3_values, label='IC_p3', marker='^', color='pink')
    plt.plot(factors[min_ic_p3_idx], IC_p3_values[min_ic_p3_idx],
             marker='^', color='purple', markersize=10, label=f'Min IC_p3')

    plt.xlabel('Number of Factors')
    plt.ylabel('Information Criteria')
    plt.title('Bai & Ng Information Criteria vs Number of Factors')
    plt.legend()
    plt.grid(True)
    plt.xticks(factors)

    plt.show()

# Example usage (assuming X and max_factors are defined):
# plot_bai_ng_criteria(X, max_factors)

def plot_eigenvalues(values):
    """
    Plot eigenvalues from PCA in a simple bar chart using Plotly.

    Parameters:
    - pca: Fitted PCA object with explained_variance_ attribute
    """
    n_components = len(values)

    fig = go.Figure([go.Bar(x=list(range(1, n_components + 1)),
                            y=values,
                            name='Eigenvalues',
                            marker_color='skyblue',
                            hovertemplate='Component %{x}<br>Eigenvalue: %{y:.4f}')])

    fig.update_layout(title_text="Eigenvalues of Principal Components",
                      xaxis_title="Principal Component Number",
                      yaxis_title="Eigenvalue",
                      template='plotly_white',
                      width=600,
                      height=400)

    fig.show()

# Data loading and preparation
df = pd.read_csv('preprocessed_current.csv', index_col=0, parse_dates=True)
X = df.values # T x n

max_factors = 30

pca = PCA(n_components=max_factors)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

transformed_X = pca.fit_transform(X_scaled)
eigenvalues = pca.explained_variance_
eigenvalue_ratios = pca.explained_variance_ratio_

log_eigenvalues = np.log(eigenvalues)
print(eigenvalue_ratios)

print("ER results:", er_test(eigenvalue_ratios))

print("GR results:", gr_test(eigenvalues))

plot_eigenvalues(eigenvalues)

ai1, ai2, ai3 = bai_ng_criteria(X_scaled, max_factors)

plot_bai_ng_criteria(X_scaled, max_factors)

print("ai1, factor value:",np.argmin(ai1))
print("ai2, factor value:",np.argmin(ai2))
print("ai3, factor value:",np.argmin(ai3))
