
'''
Adjust mean to 0
Find Covar Matrix
Compute the Eigenvlaues and Eigenvectors of the Covar Matrix
Order Eigenvalues from highest to lowest
Select the first few eigen values
Generate a new representation of the data

'''

import pandas as pd
import numpy as np

df = pd.read_csv('current.csv', index_col='sasdate', parse_dates=True)

print(df)