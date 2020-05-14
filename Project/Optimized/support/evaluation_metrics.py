import numpy as np

def RMSE(test_data, predicted):
    """Calculate Root Mean Square Error by ignoring missing values in the test data."""
    I = ~np.isnan(test_data)  # indicator for missing values    
    sqerror = abs(test_data - predicted) ** 2  # squared error array
    mse = np.nanmean(sqerror[I])  # mean squared error
    return np.sqrt(mse)


def MAE(test_data, predicted):
    """Calculate Mean Absolute Error by ignoring missing values in the test data."""
    I = ~np.isnan(test_data)    
    error = abs(test_data - predicted)  # squared error array
    mae = np.nanmean(error[I])  # mean error
    return mae
