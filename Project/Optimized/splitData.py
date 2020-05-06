import pandas as pd
import numpy as np

# Split the data into train(80%) and test(20%)
def split_train_test(data, percent_test=0.2):
    n, m = data.shape             # # users, # movies
    N = n * m                     # # cells in matrix

    # Preparing train/test ndarrays.
    train = data.copy()
    test = np.ones(data.shape) * np.nan

    # Drawing random sample of training data to use for testing.
    tosample = np.where(~np.isnan(train))       # ignore nan values in data
    idx_pairs = list(zip(tosample[0], tosample[1]))   # tuples of row/col index pairs

    test_size = int(len(idx_pairs) * percent_test)  # use 20% of data as test set
    train_size = len(idx_pairs) - test_size   # and remainder for training

    indices = np.arange(len(idx_pairs))         # indices of index pairs
    sample = np.random.choice(indices, replace=False, size=test_size)

    # Transfering random sample from train set to test set.
    for idx in sample:
        idx_pair = idx_pairs[idx]
        test[idx_pair] = train[idx_pair]  # transfer to test set
        train[idx_pair] = np.nan          # remove from train set

    # Verifying everything worked properly
    assert(train_size == N-np.isnan(train).sum())
    assert(test_size == N-np.isnan(test).sum())

    # Returning train set and test set
    return train, test
