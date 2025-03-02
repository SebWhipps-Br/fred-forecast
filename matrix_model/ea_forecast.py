import numpy as np

data = np.load('processed_data/processed_3d_data.npz')
array_3d = data['data']        # Numeric 3D array
countries = data['countries']  # List of country codes
variables = data['variables']  # List of variable names (no postfixes)
dates = data.get('dates', None)  # Dates array

print(f"Shape: {array_3d.shape} (countries × time × variables)")
