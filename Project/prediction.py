# Support Functions for prediction

import numpy as np
import pandas as pd

# This function receives the correlation values and a list of ratings, if  the rating is different than Nan
# the we multiply  the value of the correlations by the rating otherwise no. The sum of all the operations is returned.

def sumproduct(correlations, ratings, user_mean):
    sum = 0
    sum = np.nansum(correlations*(ratings - user_mean))
    return sum

# This function receives the correlation values and a list of ratings, if  the rating is different than Nan
# the we sum up the value of the correlations, otherwise no. The value of the sum is returned


def sumif(correlations, ratings):
    # sum all correlation values for users who have rated the item before
    sum = 0
    ratings_notnan = np.isfinite(ratings)
    sum = np.nansum(correlations * ratings_notnan)
    return sum

# Implementation of the Normalized score rating function, this is implemented  using the correlation
# the ratings and the mean values of the users


def prediction_normalized(correlations, ratings, k, user_mean, ru_mean):
    ratings = np.array(ratings, dtype=float)
    # k is the neigborhood size
    if correlations.shape[1] > 0 and sumif(correlations.values.reshape(k), ratings) > 0:
        return ru_mean + (sumproduct(correlations.values.reshape(k), ratings, user_mean) / sumif(correlations.values.reshape(k), ratings))
    return np.nan