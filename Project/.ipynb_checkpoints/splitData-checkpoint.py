import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_train_test(data, percent_test=0.1):
    train, test = train_test_split(data, test_size= percent_test)

    train = train.pivot_table(index='movieId', columns="userId", values="movieRating")
    test = test.pivot_table(index='movieId', columns="userId", values="movieRating")

    # Return train set and test set
    return train, test
