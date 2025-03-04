import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


class MatrixFactorModel:
    def __init__(self, X, k1, k2, max_iterations=1):
        """
        Initialize the matrix factor model with standardized data.

        Parameters:
        - X: 3D array of shape (T, p1, p2) - time x rows x columns
        - k1: Number of row factors (for R)
        - k2: Number of column factors (for C)
        - max_iterations: Number of projection iterations (default 1)
        """
        self.X_unscaled = X  # Keeps original
        self.X = self.standardise_data(X) # Scales the data
        self.T, self.p1, self.p2 = X.shape  # time, rows (countries), columns (variables)
        self.k1 = k1  # Row factors
        self.k2 = k2  # Column factors
        self.max_iterations = max_iterations

        # Initialise loading matrices
        self.R = None  # Row loading matrix (p1 x k1)
        self.C = None  # Column loading matrix (p2 x k2)

    def standardise_data(self, X):
        """
        Standardises the input data along the time axis (mean 0, std 1).

        Parameters:
        - X: 3D array of shape (T, p1, p2)

        Returns:
        - X_scaled: Standardised array of same shape
        """
        X_scaled = X.copy().astype(float)  # Avoids modifying original, ensure float
        mean = np.mean(X_scaled, axis=0, keepdims=True)  # Shape: (1, p1, p2)
        std = np.std(X_scaled, axis=0, keepdims=True)  # Shape: (1, p1, p2)
        X_scaled = (X_scaled - mean) / std
        return X_scaled

    def compute_M(self, axis='rows'):
        """
        Compute the scaled sample covariance matrix M̂.

        Parameters:
            - axis: axis along which to compute the covariance matrix; either 'rows' or 'cols'
        """
        if axis == 'rows':
            M = np.zeros((self.p1, self.p1))  # p1 x p1 (e.g., 8 x 8)
            for t in range(self.T):
                Xt = self.X[t]  # p1 x p2
                M += Xt @ Xt.T
            M /= (self.T * self.p1 * self.p2)
        elif axis == 'columns':
            M = np.zeros((self.p2, self.p2))  # p2 x p2 (e.g., 37 x 37)
            for t in range(self.T):
                Xt = self.X[t]  # p1 x p2
                M += Xt.T @ Xt
            M /= (self.T * self.p1 * self.p2)
        return M

    def initial_estimate(self):
        """Computes initial estimates R̂ and Ĉ using PCA."""
        M1 = self.compute_M(axis='rows')  # p1 x p1
        pca1 = PCA(n_components=self.k1)
        pca1.fit(M1)
        Q1 = pca1.components_.T
        self.R = np.sqrt(self.p1) * Q1
        print(f"Initial R shape: {self.R.shape}, Explained variance: {pca1.explained_variance_ratio_}")

        M2 = self.compute_M(axis='columns')  # p2 x p2
        pca2 = PCA(n_components=self.k2)
        pca2.fit(M2)
        Q2 = pca2.components_.T
        self.C = np.sqrt(self.p2) * Q2
        print(f"Initial C shape: {self.C.shape}, Explained variance: {pca2.explained_variance_ratio_}")

    def project_data(self):
        """Project X into lower-dimensional Ŷ_t and Ẑ_t."""
        Y_hat = np.zeros((self.T, self.p1, self.k2))
        Z_hat = np.zeros((self.T, self.p2, self.k1))
        for t in range(self.T):
            Xt = self.X[t]
            Y_hat[t] = (Xt @ self.C) / self.p2  # Ŷ_t
            Z_hat[t] = (Xt.T @ self.R) / self.p1  # Ẑ_t
        return Y_hat, Z_hat

    def refine_estimates(self, Y_hat, Z_hat):
        """Refine R and C using projected data."""
        M1_tilde = np.zeros((self.p1, self.p1))
        for t in range(self.T):
            Yt = Y_hat[t]
            M1_tilde += Yt @ Yt.T
        M1_tilde /= (self.T * self.p1)
        pca1 = PCA(n_components=self.k1)
        pca1.fit(M1_tilde)
        Q1_tilde = pca1.components_.T
        self.R = np.sqrt(self.p1) * Q1_tilde
        print(f"Refined R shape: {self.R.shape}, Explained variance: {pca1.explained_variance_ratio_}")

        M2_tilde = np.zeros((self.p2, self.p2))
        for t in range(self.T):
            Zt = Z_hat[t]
            M2_tilde += Zt @ Zt.T
        M2_tilde /= (self.T * self.p2)
        pca2 = PCA(n_components=self.k2)
        pca2.fit(M2_tilde)
        Q2_tilde = pca2.components_.T
        self.C = np.sqrt(self.p2) * Q2_tilde
        print(f"Refined C shape: {self.C.shape}, Explained variance: {pca2.explained_variance_ratio_}")

    def fit(self):
        """Fits the matrix factor model."""
        print("Computing initial estimates...")
        self.initial_estimate()
        for i in range(self.max_iterations):
            print(f"Iteration {i + 1}/{self.max_iterations}")
            Y_hat, Z_hat = self.project_data()
            self.refine_estimates(Y_hat, Z_hat)

    def get_loading_matrices(self):
        """Return the estimated R and C."""
        return self.R, self.C

    def estimate_factors(self):
        """Estimate F_t for each time t."""
        R_pinv = np.linalg.pinv(self.R)  # Shape: (k1, p1)
        C_pinv = np.linalg.pinv(self.C)  # Shape: (k2, p2)
        F = np.zeros((self.T, self.k1, self.k2))  # T x k1 x k2
        for t in range(self.T):
            Xt = self.X[t]  # p1 x p2
            F[t] = R_pinv @ Xt @ C_pinv.T  # (k1 x p1) @ (p1 x p2) @ (p2 x k2) = (k1 x k2)
        return F

    def reconstruct_X(self):
        """Reconstruct X̂_t = R F_t C^T for each t."""
        F = self.estimate_factors()  # T x k1 x k2
        X_hat = np.zeros((self.T, self.p1, self.p2))
        C_T = self.C.T  # k2 x p2
        for t in range(self.T):
            X_hat[t] = self.R @ F[t] @ C_T  # (p1 x k1) @ (k1 x k2) @ (k2 x p2) = (p1 x p2)
        return X_hat

    def compute_residuals(self):
        """Compute residuals E_t = X_t - X̂_t."""
        X_hat = self.reconstruct_X()
        E = self.X - X_hat  # T x p1 x p2
        return E

    def check_model(self):
        """Check model fit with diagnostics."""
        X_hat = self.reconstruct_X()
        E = self.compute_residuals()

        # Frobenius norm of residuals per time point
        residual_norms = np.linalg.norm(E, ord='fro', axis=(1, 2))  # Shape: (T,)
        avg_residual_norm = np.mean(residual_norms)

        # Explained variance ratio
        total_variance = np.sum(np.linalg.norm(self.X, ord='fro', axis=(1, 2)) ** 2)
        explained_variance = np.sum(np.linalg.norm(X_hat, ord='fro', axis=(1, 2)) ** 2)
        residual_variance = np.sum(np.linalg.norm(E, ord='fro', axis=(1, 2)) ** 2)
        explained_variance_ratio = explained_variance / total_variance

        print(f"\nModel Diagnostics:")
        print(f"Average Frobenius norm of residuals: {avg_residual_norm:.4f}")
        print(f"Explained variance ratio: {explained_variance_ratio:.4f}")
        print(f"Residual variance ratio: {residual_variance / total_variance:.4f}")

        return X_hat, E, residual_norms

    def plot_variable_across_countries(self, variable_idx, country_names=None):
        """
        Plot a variable across all countries, comparing original and model data.

        Parameters:
        - variable_idx: Index of the variable to plot (0 to p2-1)
        - country_names: List of country names (optional, defaults to indices)
        """
        if variable_idx < 0 or variable_idx >= self.p2:
            raise ValueError(f"variable_idx must be between 0 and {self.p2 - 1}")

        # Get original and reconstructed data for the variable
        X_orig = self.X_unscaled[:, :, variable_idx]  # Shape: (T, p1)
        X_hat = self.reconstruct_X()[:, :, variable_idx]  # Shape: (T, p1)

        # Reverse standardization for X_hat to match X_unscaled
        mean = np.mean(self.X_unscaled[:, :, variable_idx], axis=0, keepdims=True)
        std = np.std(self.X_unscaled[:, :, variable_idx], axis=0, keepdims=True)
        X_hat_unscaled = X_hat * std + mean  # Shape: (T, p1)

        # Set up country names
        if country_names is None or len(country_names) != self.p1:
            country_names = [f"Country {i}" for i in range(self.p1)]

        # Plot
        plt.figure(figsize=(12, 6))
        for i in range(self.p1):
            plt.plot(X_orig[:, i], label=f"{country_names[i]} (Original)", alpha=0.7)
            plt.plot(X_hat_unscaled[:, i], '--', label=f"{country_names[i]} (Model)", alpha=0.7)
        plt.title(f"Variable {variable_idx} Across Countries")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


# Load and run
data = np.load('processed_data/processed_3d_data.npz')
X_unscaled = data['data']  # Shape: (257, 8, 37)
countries = data['countries']
variables = data['variables']
dates = data.get('dates', None)

print(f"Shape: {X_unscaled.shape} (time × countries × variables)")
T, n_countries, n_variables = X_unscaled.shape

COUNTRY_FACTORS = 3
VARIABLE_FACTORS = 5

# Fits the model
model = MatrixFactorModel(X_unscaled, k1=COUNTRY_FACTORS, k2=VARIABLE_FACTORS, max_iterations=1)
model.fit()

# Evaluates model
R, C = model.get_loading_matrices()
print("\nFinal R (countries x factors):\n", R[:5, :])
print("Final C (variables x factors):\n", C[:5, :])
X_hat, E, residual_norms = model.check_model()

model.plot_variable_across_countries(variable_idx=0, country_names=countries)