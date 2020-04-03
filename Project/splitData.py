import numpy as np
import pandas as pd


def split_train_test(data, percent_test=0.1):
    data = data
    """Split the data into train/test sets.
    :param int percent_test: Percentage of data to use for testing. Default 10.
    """
    n, m = data.shape             # # users, # movies
    N = n * m                     # # cells in matrix

    # Prepare train/test ndarrays.
    train = data.copy()
    test = pd.DataFrame(columns=data.columns, index=data.index)

    # Draw random sample of training data to use for testing.
    idx_pairs = train.where(~pd.isnull(train)).stack().index.values

    # use 20% of data as test set
    test_size = int(len(idx_pairs) * percent_test)
    train_size = len(idx_pairs) - test_size   # and remainder for training

    indices = np.arange(len(idx_pairs))         # indices of index pairs
    sample = np.random.choice(indices, replace=False, size=test_size)

    # Transfer random sample from train set to test set.
    for idx in sample:
        id_item = idx_pairs[idx]
        test[id_item[1]][id_item[0]] = train[id_item[1]
                                             ][id_item[0]]  # transfer to test set
        train[id_item[1]][id_item[0]] = np.nan          # remove from train set

    # Verify everything worked properly
    assert(train_size == N-(pd.isnull(train).sum()).sum())
    assert(test_size == N-(pd.isnull(test).sum()).sum())

    # Return train set and test set
    return train, test
