import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


def estimate_spectral_density(eigenvalues, eigenvectors, n, theta_points):
    """
    Estimate Σ_χ(θ) for a set of frequencies theta_points using pre-computed PCA results.

    Parameters:
    - eigenvalues: Array of eigenvalues from PCA, sorted by magnitude in descending order
    - eigenvectors: Matrix of eigenvectors from PCA where columns correspond to eigenvalues
    - n: Number of variables
    - theta_points: Array of frequencies from -π to π

    Returns:
    - sigma_chi_theta: List of estimated Σ_χ(θ) for each θ
    """
    q = len(eigenvalues)  # Assuming you've selected the top q components already

    # Step 1: Compute the spectral density matrix for each θ in theta_points
    sigma_chi_theta = []
    for theta in theta_points:
        # Compute Σ_χ(θ) = Σ_{j=1}^q λ_j(θ) * p_j(θ) * p_j(θ)^H where p_j(θ) are eigenvectors
        sigma_chi = np.zeros((n, n), dtype=complex)
        for j in range(q):
            p_j = eigenvectors[:, j]
            sigma_chi += eigenvalues[j] * np.outer(p_j, np.conj(p_j))
        sigma_chi_theta.append(sigma_chi)

    return sigma_chi_theta


def compute_autocovariance(sigma_chi_theta, theta_points, h):
    """
    Compute Γ_χ(h) from Σ_χ(θ).

    Parameters:
    - sigma_chi_theta: List of spectral density matrices for each theta
    - theta_points: Array of frequencies from -π to π
    - h: The lag for which to compute autocovariance

    Returns:
    - gamma_chi_h: The autocovariance matrix at lag h
    """
    gamma_chi_h = np.zeros_like(sigma_chi_theta[0])
    for sigma, theta in zip(sigma_chi_theta, theta_points):
        gamma_chi_h += sigma.real * np.cos(h * theta) - sigma.imag * np.sin(h * theta)

    # Normalize by the number of frequency points
    return gamma_chi_h / len(theta_points)


def forecast_common_components(xt, h, q, n, T, num_theta_points=100):
    """
    Forecast the common components χT+h|T using the formula:
    χ_T+h|T = Γχ(h) Q (Q' Γχ(0) Q)^(-1) Q' xT

    Parameters:
    - xt: np.array of shape (T, n), observed data
    - h: int, forecast horizon
    - q: int, number of factors
    - n: int, number of variables
    - T: int, number of time periods

    Returns:
    - chi_forecast: np.array, h-step-ahead forecast of common components
    """

    pca = PCA(n_components=q)
    F_hat = pca.fit_transform(xt)  # The estimated factors
    Q_hat = pca.components_ # The loadings
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_.T

    theta_points = np.linspace(-np.pi, np.pi, num_theta_points)  # Discretise frequency domain - num can be increased at time cost
    sigma_chi_theta = estimate_spectral_density(eigenvalues,eigenvectors, n, theta_points)

    Gamma_chi_h = compute_autocovariance(sigma_chi_theta, theta_points, h)
    Gamma_chi_0 = compute_autocovariance(sigma_chi_theta, theta_points, 0)

    x_last = xt[-1]
    #print('Q_hat:', Q_hat.shape, 'Gamma_chi_0', Gamma_chi_0.shape, 'Q_hat.T', Q_hat.T.shape)
    pre_inv_term = Q_hat @ Gamma_chi_0 @ Q_hat.T

    if np.linalg.cond(pre_inv_term) < 1 / np.finfo(pre_inv_term.dtype).eps: # the matrix is well-conditioned for standard inversion
        inv_term = np.linalg.inv(pre_inv_term)
    else:
        inv_term = np.linalg.pinv(pre_inv_term)


    #print('Gamma_chi_h:', Gamma_chi_h.shape, 'Q_hat', Q_hat.shape, 'inv_term', inv_term.shape, 'Q_hat.T', Q_hat.T.shape,'x_last', x_last.shape)

    chi_forecast = np.real((Gamma_chi_h @ Q_hat.T @ inv_term) @ (Q_hat @ x_last))
    return chi_forecast

df = pd.read_csv('preprocessed_current.csv', index_col=0, parse_dates=True)
X = df.values # Converts to numpy array

T, n = X.shape # T - the number of data points, n - dimensionality


# Separating the last month for actual values and using the rest for training
h = 1 # Forecasting just 1 month ahead
T -= h # Adjusting T as we removed a month
X_train = X[:-h]  # All data except the last month
X_actual = X[-h]  # Last month's data as actual values
q = 10  # the number of latent factors
print("T:", T, "n:", n)


chi_forecast = forecast_common_components(X_train, h, q, n, T)
print(f"Forecast for common components at T+{h}: {chi_forecast}")

mse = mean_squared_error(X_actual, chi_forecast.flatten())
print(f'Overall Mean Squared Error: {mse}')

# For individual variables:
for i in range(n):
    mse_i = mean_squared_error([X_actual[i]], [chi_forecast[i]])
    print(f'MSE for variable {df.columns[i]}: {mse_i}')