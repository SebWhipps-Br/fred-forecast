# File: matrix_factor_analysis.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def bai_ng_criteria_matrix(X, max_k1, max_k2, axis='rows'):
    """
    Compute Bai & Ng Information Criteria (IC_p1, IC_p2, IC_p3) for matrix factor models.

    Parameters:
    - X: Numpy array of shape (T, p1, p2) where T is time, p1 is rows (countries), p2 is columns (variables)
    - max_k1: Integer, maximum number of row factors
    - max_k2: Integer, maximum number of column factors
    - axis: str, 'rows' for country factors (R), 'columns' for variable factors (C)

    Returns:
    - Tuple (IC_p1_values, IC_p2_values, IC_p3_values) for the specified axis
    """
    T, p1, p2 = X.shape
    n = p1 if axis == 'rows' else p2  # Number of entities (countries or variables)
    max_factors = max_k1 if axis == 'rows' else max_k2

    IC_p1_values, IC_p2_values, IC_p3_values = [], [], []

    # Compute covariance matrix
    if axis == 'rows':
        M = np.zeros((p1, p1))
        for t in range(T):
            Xt = X[t]  # p1 x p2
            M += Xt @ Xt.T
        M /= (T * p1 * p2)
    else:  # 'columns'
        M = np.zeros((p2, p2))
        for t in range(T):
            Xt = X[t]
            M += Xt.T @ Xt
        M /= (T * p1 * p2)

    for k in range(1, max_factors + 1):
        # Perform PCA
        pca = PCA(n_components=k)
        pca.fit(M)
        loadings = pca.components_.T  # Shape: (n, k)
        factors = pca.transform(M)    # Shape: (n, k)

        # Reconstruct M and compute residuals
        M_hat = factors @ loadings.T  # Shape: (n, n)
        residuals = M - M_hat

        # Log of residual variance
        ln_V_k_F = np.log(np.sum(residuals ** 2) / (n * n))

        # Compute criteria terms
        term1 = np.log((n * T) / (n + T))
        term2 = (n + T) / (n * T)
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
    """
    Eigenvalue Ratio (ER) test to determine the number of factors.

    Parameters:
    - ratios: Array of eigenvalue ratios (lambda_i / lambda_{i+1})

    Returns:
    - int, number of factors based on maximum ratio
    """
    if len(ratios) < 1:
        return 1
    max_ratio_idx = np.argmax(ratios)
    return max_ratio_idx + 1


def gr_test(eigenvalues):
    """
    Growth Ratio (GR) test to determine the number of factors.

    Parameters:
    - eigenvalues: Array of eigenvalues from PCA

    Returns:
    - int, number of factors based on growth ratio drop
    """
    growth_rates = [eigenvalues[i] / eigenvalues[i + 1] for i in range(len(eigenvalues) - 1)]
    print("Growth rates:", growth_rates)
    gr_ratios = [growth_rates[i] / growth_rates[i - 1] if i > 0 else 0 for i in range(1, len(growth_rates))]
    max_gr_drop_idx = np.argmin(gr_ratios)
    return max_gr_drop_idx + 1


def plot_bai_ng_criteria_matrix(X, max_k1, max_k2, axis='rows', save_path_prefix='figures/bai_ng_ic'):
    """
    Plot Bai & Ng Information Criteria for matrix factor models.

    Parameters:
    - X: Numpy array of shape (T, p1, p2)
    - max_k1: Integer, maximum number of row factors
    - max_k2: Integer, maximum number of column factors
    - axis: str, 'rows' or 'columns'
    - save_path_prefix: str, prefix for saving plot
    """
    IC_p1_values, IC_p2_values, IC_p3_values = bai_ng_criteria_matrix(X, max_k1, max_k2, axis)
    factors = list(range(1, len(IC_p1_values) + 1))
    min_ic_p1_idx = np.argmin(IC_p1_values)
    min_ic_p2_idx = np.argmin(IC_p2_values)
    min_ic_p3_idx = np.argmin(IC_p3_values)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(factors, IC_p1_values, label='IC_p1', marker='o', color='blue')
    plt.plot(factors[min_ic_p1_idx], IC_p1_values[min_ic_p1_idx], marker='o', color='red',
             markersize=10, label=f'Min IC_p1 (k={min_ic_p1_idx + 1})')
    plt.plot(factors, IC_p2_values, label='IC_p2', marker='s', color='orange')
    plt.plot(factors[min_ic_p2_idx], IC_p2_values[min_ic_p2_idx], marker='s', color='green',
             markersize=10, label=f'Min IC_p2 (k={min_ic_p2_idx + 1})')
    plt.plot(factors, IC_p3_values, label='IC_p3', marker='^', color='pink')
    plt.plot(factors[min_ic_p3_idx], IC_p3_values[min_ic_p3_idx], marker='^', color='purple',
             markersize=10, label=f'Min IC_p3 (k={min_ic_p3_idx + 1})')

    plt.xlabel('Number of Factors', fontsize=14)
    plt.ylabel('Information Criteria', fontsize=14)
    plt.title(f'Bai & Ng Criteria for {"Country" if axis == "rows" else "Variable"} Factors', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(factors, fontsize=12)
    plt.tight_layout()

    save_path = f'{save_path_prefix}_{axis}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_eigenvalues(values, title, save_path):
    """
    Plot eigenvalues as a high-resolution bar chart using Matplotlib.

    Parameters:
    - values: Array of eigenvalues
    - title: str, plot title
    - save_path: str, file path to save the plot
    """
    n_components = len(values)
    components = list(range(1, n_components + 1))

    plt.figure(figsize=(12, 8), dpi=300)
    plt.bar(components, values, color='skyblue', edgecolor='black', width=0.8)
    plt.xlabel('Principal Component Number', fontsize=14)
    plt.ylabel('Eigenvalue', fontsize=14)
    plt.title(title, fontsize=16, pad=10)
    plt.xticks(components, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_eigenvalues_plotly(values, title):
    """
    Plot eigenvalues using Plotly.

    Parameters:
    - values: Array of eigenvalues
    - title: str, plot title
    """
    n_components = len(values)
    fig = go.Figure([go.Bar(x=list(range(1, n_components + 1)), y=values, name='Eigenvalues',
                            marker_color='skyblue', hovertemplate='Component %{x}<br>Eigenvalue: %{y:.4f}')])
    fig.update_layout(title_text=title, xaxis_title="Principal Component Number", yaxis_title="Eigenvalue",
                      template='plotly_white', width=600, height=400)
    fig.show()


# Example usage
if __name__ == "__main__":
    # Load data
    data = np.load('processed_data/processed_3d_data.npz')
    X_unscaled = data['data']
    HORIZON = 12
    X = X_unscaled[:-HORIZON]  # Training data only
    T, p1, p2 = X.shape

    # Standardize data (replicating MatrixFactorModel's _standardise_data)
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    X_scaled = (X - mean) / std

    # Parameters
    MAX_K1 = 5  # Max country factors
    MAX_K2 = 10  # Max variable factors

    # Compute covariance matrices
    M_rows = np.zeros((p1, p1))
    M_cols = np.zeros((p2, p2))
    for t in range(T):
        Xt = X_scaled[t]
        M_rows += Xt @ Xt.T
        M_cols += Xt.T @ Xt
    M_rows /= (T * p1 * p2)
    M_cols /= (T * p1 * p2)

    # PCA for eigenvalues
    pca_rows = PCA(n_components=min(MAX_K1, p1))
    pca_cols = PCA(n_components=min(MAX_K2, p2))
    pca_rows.fit(M_rows)
    pca_cols.fit(M_cols)

    row_eigenvalues = pca_rows.explained_variance_
    col_eigenvalues = pca_cols.explained_variance_
    row_ratios = row_eigenvalues[:-1] / row_eigenvalues[1:]
    col_ratios = col_eigenvalues[:-1] / col_eigenvalues[1:]

    # Factor selection
    print("Row Factors (Countries):")
    print("ER results:", er_test(row_ratios))
    print("GR results:", gr_test(row_eigenvalues))
    print("\nColumn Factors (Variables):")
    print("ER results:", er_test(col_ratios))
    print("GR results:", gr_test(col_eigenvalues))

    # Bai & Ng Criteria
    ic1_rows, ic2_rows, ic3_rows = bai_ng_criteria_matrix(X_scaled, MAX_K1, MAX_K2, axis='rows')
    ic1_cols, ic2_cols, ic3_cols = bai_ng_criteria_matrix(X_scaled, MAX_K1, MAX_K2, axis='columns')
    print("\nBai & Ng Results (Rows):")
    print("IC_p1 min at k:", np.argmin(ic1_rows) + 1)
    print("IC_p2 min at k:", np.argmin(ic2_rows) + 1)
    print("IC_p3 min at k:", np.argmin(ic3_rows) + 1)
    print("\nBai & Ng Results (Columns):")
    print("IC_p1 min at k:", np.argmin(ic1_cols) + 1)
    print("IC_p2 min at k:", np.argmin(ic2_cols) + 1)
    print("IC_p3 min at k:", np.argmin(ic3_cols) + 1)

    # Plotting
    plot_bai_ng_criteria_matrix(X_scaled, MAX_K1, MAX_K2, axis='rows')
    plot_bai_ng_criteria_matrix(X_scaled, MAX_K1, MAX_K2, axis='columns')
    plot_eigenvalues(row_eigenvalues, 'Eigenvalues for Country Factors',
                     'figures/eigenvalues_country.png')
    plot_eigenvalues(col_eigenvalues, 'Eigenvalues for Variable Factors',
                     'figures/eigenvalues_variable.png')
    plot_eigenvalues_plotly(row_eigenvalues, 'Eigenvalues for Country Factors')
    plot_eigenvalues_plotly(col_eigenvalues, 'Eigenvalues for Variable Factors')