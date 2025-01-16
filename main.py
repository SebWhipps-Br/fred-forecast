
'''
Find Covar Matrix
Compute the Eigenvlaues and Eigenvectors of the Covar Matrix
Order Eigenvalues from highest to lowest
Select the first few eigen values
Generate a new representation of the data

'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('preprocessed_current.csv', index_col=0, parse_dates=True)

print(df)

X = df.values # convert to numpy array

# Determine number of observations and variables
T, n = X.shape

# Separate the last month for actual values and use the rest for training
X_train = X[:-1]  # All data except the last month
X_actual = X[-1]  # Last month's data as actual values

# Assuming you want to forecast one step ahead (h=1)
h = 1

# Adjust T since we've removed one observation
T -= 1

# Estimate Factors Using PCA:
n_components = 5  # Example value, adjust based on your model selection criteria

pca = PCA(n_components=n_components)
F = pca.fit_transform(X_train)  # F now contains the factors for each time period minus the last month

# Estimate Loadings (`β`):
beta = []

# For each variable, perform regression on the factors
for i in range(n):
    model = LinearRegression()
    model.fit(F[:-h], X_train[h:, i])  # Use all but last h observations of F and X_train
    beta.append(model.coef_)

beta = np.array(beta)

# Make Forecasts:
# Use the last observation of factors from training to forecast for T+1
X_forecast = np.dot(beta, F[-1])

# Ensure X_forecast has the same shape as X_actual
if X_forecast.ndim == 1:
    X_forecast = X_forecast.reshape(1, -1)  # Reshape if it's a 1D array

# Evaluation:
# Calculate MSE for this forecast
print(X_actual.shape)
print(X_forecast.flatten().shape)

mse = mean_squared_error(X_actual, X_forecast.flatten())
print(f'Overall Mean Squared Error: {mse}')

# For individual variables:
for i in range(n):
    mse_i = mean_squared_error([X_actual[i]], [X_forecast[0, i]])  # Index with 0 for the single forecast
    print(f'MSE for variable {df.columns[i]}: {mse_i}')

print("\nComparison of Actual vs Predicted Values:")
for i, (actual, predicted) in enumerate(zip(X_actual, X_forecast.flatten())):
    print(f"{df.columns[i]:<15} | Actual: {actual:.4f} | Predicted: {predicted:.4f}")