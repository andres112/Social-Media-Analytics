import numpy as np
import pandas as pd

# Define our evaluation function. 
def get_rmse(test_data, predicted): 
    """Calculate root mean squared error.
    Ignoring missing values in the test data. """
    I = ~pd.isnull(test_data) # indicator for missing values 
    N = I.values.sum()
    # number of non-missing values
    sqerror = abs(test_data - predicted) ** 2 # squared error array 
    mse = (sqerror[I].sum() / N ).sum()
    return np.sqrt(mse)